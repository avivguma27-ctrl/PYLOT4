import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from pytrends.request import TrendReq
import praw
import xgboost as xgb
from transformers import pipeline
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import yfinance as yf
from telegram import Bot

# ============================== #
#         קונפיג קשיח (עם טוקנים) #
# ============================== #

DB_FILE = "penny_stocks.db"
LOG_FILE = "bot.log"
OUTPUT_FILE = "top_stocks.csv"

# 🔑 מפתחות / טוקנים (כפי שנתת)
TELEGRAM_TOKEN = "8453354058:AAGG0v0zLWTe1NJE7ttfaUZvoutf5XNGU7s"
CHAT_ID = "6387878532"
FMP_API_KEY = "5nhxZGIiFnjG8JxcdSKljx0eZRuqwELX"
REDDIT_CLIENT_ID = "ZOa0YjqoW-H_-aFXhIXrLw"
REDDIT_CLIENT_SECRET = "7v6s4PJr2kdbvtfNDq7khltKXVkCrw"
REDDIT_USER_AGENT = "_bot_v1"

# ספים ובקרים
GAIN_THRESHOLD = 0.05
RSI_COLD_THRESHOLD = 40
VOLUME_THRESHOLD = 500_000
MARKET_CAP_THRESHOLD = 50_000_000
FLOAT_THRESHOLD = 50_000_000
MAX_API_RETRIES = 3
RETRY_DELAY = 15
MAX_TICKERS = 10
RATE_LIMIT_PER_MINUTE = 60
LOOKBACK = 30
MAX_CONCURRENT_REQUESTS = 5

# ============================== #
#         Logging & Bot          #
# ============================== #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

bot = Bot(token=TELEGRAM_TOKEN)

# ============================== #
#          DB + Logging          #
# ============================== #
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                score REAL,
                predicted_gain REAL,
                days_in_trade INTEGER,
                position_size REAL,
                timestamp TEXT,
                google_trend REAL,
                reddit_sentiment REAL,
                short_interest REAL,
                feature_importance TEXT,
                UNIQUE(ticker, timestamp)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_message TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()

def log_error(msg: str):
    logging.error(msg)
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO errors (error_message, timestamp) VALUES (?, ?)",
                (msg, datetime.utcnow().isoformat()),
            )
            conn.commit()
    except Exception as e:
        logging.error(f"DB log_error failed: {e}")

async def send_telegram(msg: str):
    """שליחת הודעה לטלגרם עם נסיונות חוזרים (בלי רקורסיה)."""
    for attempt in range(MAX_API_RETRIES):
        try:
            await bot.send_message(chat_id=CHAT_ID, text=msg)
            return
        except Exception as e:
            log_error(f"Telegram error (attempt {attempt + 1}): {e}")
            if attempt < MAX_API_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    log_error("Failed to send Telegram message after retries")

# ============================== #
#         Helpers: TA/ML         #
# ============================== #
def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = (df["High"] - df["Low"]).abs()
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ============================== #
#      Google Trends + Reddit     #
# ============================== #
_sentiment_analyzer = None
def get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english'
        )
    return _sentiment_analyzer

async def get_google_trends(ticker: str) -> float:
    for attempt in range(MAX_API_RETRIES):
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload([ticker], timeframe='now 7-d')
            data = pytrends.interest_over_time()
            if not data.empty and ticker in data.columns:
                return float(data[ticker].mean() / 100)
            await asyncio.sleep(60 / RATE_LIMIT_PER_MINUTE)
        except Exception as e:
            log_error(f"Google Trends error for {ticker} (attempt {attempt + 1}): {e}")
            if attempt < MAX_API_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    return 0.0

async def analyze_reddit_sentiment(ticker: str) -> float:
    for attempt in range(MAX_API_RETRIES):
        try:
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            analyzer = get_sentiment_analyzer()
            sentiment = 0.0
            count = 0
            for post in reddit.subreddit("wallstreetbets+pennystocks").search(ticker, limit=5):
                text = (post.title or '') + ' ' + (getattr(post, 'selftext', '') or '')
                res = analyzer(text[:512])[0]
                score = float(res['score']) if res['label'] == 'POSITIVE' else -float(res['score'])
                sentiment += score
                count += 1
                await asyncio.sleep(60 / RATE_LIMIT_PER_MINUTE)
            return sentiment / count if count > 0 else 0.0
        except Exception as e:
            log_error(f"Reddit sentiment error for {ticker} (attempt {attempt + 1}): {e}")
            if attempt < MAX_API_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    return 0.0

# ============================== #
#         FMP / Tickers          #
# ============================== #
async def check_fmp_api() -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={FMP_API_KEY}"
            async with session.get(url, timeout=15) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return bool(data)
    except Exception as e:
        log_error(f"FMP API check failed: {e}")
        return False

async def fetch_tickers() -> List[str]:
    tickers: List[str] = []
    headers = {'User-Agent': 'Mozilla/5.0'}

    async with aiohttp.ClientSession(headers=headers) as session:
        async def _try_fmp_list():
            try:
                url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={FMP_API_KEY}"
                async with session.get(url, timeout=30) as response:
                    response.raise_for_status()
                    data = await response.json()
                df = pd.DataFrame(data)
                needed_cols = {'symbol', 'price', 'exchangeShortName'}
                if not needed_cols.issubset(df.columns):
                    raise ValueError("FMP list missing required columns")
                vol_col = next((c for c in ['volume', 'avgVolume', 'averageVolume'] if c in df.columns), None)
                if not vol_col:
                    raise ValueError("No volume column in FMP list")
                df = df.dropna(subset=['symbol', 'price', vol_col, 'exchangeShortName'])
                df = df[(df['price'] <= 5.0) &
                        (df[vol_col] >= VOLUME_THRESHOLD) &
                        (df['exchangeShortName'].isin(['NASDAQ', 'NYSE'])) &
                        (~df['symbol'].str.contains('-WS|-U|-R|-P-', na=False))]
                return df['symbol'].astype(str).tolist()
            except Exception as e:
                log_error(f"FMP list fetch error: {e}")
                return []

        async def _try_fmp_actives():
            try:
                url = f"https://financialmodelingprep.com/api/v3/stock/actives?apikey={FMP_API_KEY}"
                async with session.get(url, timeout=30) as response:
                    response.raise_for_status()
                    data = await response.json()
                arr = data.get('mostActiveStock', []) if isinstance(data, dict) else []
                rows = []
                for item in arr:
                    sym = item.get('ticker') or item.get('symbol')
                    price = item.get('price')
                    vol = item.get('volume')
                    ex = item.get('exchange') or item.get('exchangeShortName')
                    if sym and price is not None and vol is not None and ex:
                        rows.append((sym, float(price), int(vol), str(ex)))
                df = pd.DataFrame(rows, columns=['symbol','price','volume','exchange'])
                df = df[(df['price'] <= 5.0) & (df['volume'] >= VOLUME_THRESHOLD) &
                        (df['exchange'].str.contains('NAS|NY', na=False)) &
                        (~df['symbol'].str.contains('-WS|-U|-R|-P-', na=False))]
                return df['symbol'].astype(str).tolist()
            except Exception as e:
                log_error(f"FMP actives fetch error: {e}")
                return []

        have_api = await check_fmp_api()
        if have_api:
            lst = await _try_fmp_list()
            if not lst:
                lst = await _try_fmp_actives()
            tickers.extend(lst)
        else:
            log_error("FMP API unavailable; using fallback list")

        if len(tickers) < 5:
            fallback_tickers = ['AACG','AAOI','AAME','AATC','ABAT','ABCB','ABSI','ABVC','ACAD','ACET']
            tickers.extend([t for t in fallback_tickers if t not in tickers])

        tickers = list(dict.fromkeys(tickers))[:MAX_TICKERS]

    logging.info(f"Fetched {len(tickers)} tickers: {tickers}")
    return tickers

# ============================== #
#        Feature Engineering     #
# ============================== #
def prepare_features(df: pd.DataFrame, info: Dict[str, Any], vix_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = df.copy()
    df['returns_1d'] = df['Close'].pct_change()
    df['logret'] = np.log(df['Close']).diff()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['volatility_20'] = df['returns_1d'].rolling(20).std()
    df['mom_10'] = df['Close'].pct_change(10)
    df['rsi'] = rsi_series(df['Close'])
    df['atr'] = atr_series(df)

    df['short_interest'] = float(info.get('shortPercentOfFloat', 0) or 0)
    df['float'] = float(info.get('floatShares', 0) or 0)
    df['vix'] = (vix_df['Close'].iloc[-1] if vix_df is not None and not vix_df.empty else 0.0)
    df = df.dropna()

    cols = ['Close','ma_10','ma_50','ema_20','rsi','atr','returns_1d','logret','volatility_20','mom_10','short_interest','float','vix']
    return df[cols]

def timeseries_train_test(df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Predict next-day close (shift target)
    y = df_feat['Close'].shift(-1).dropna()
    X = df_feat.drop(columns=['Close']).loc[y.index]
    return X, y

# ============================== #
#          Ticker Analysis       #
# ============================== #
async def analyze_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    if not isinstance(ticker, str) or ticker.lower() == 'nan':
        log_error(f"Invalid ticker: {ticker}")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            return await asyncio.wait_for(analyze_ticker_inner(ticker, session), timeout=90)
    except asyncio.TimeoutError:
        log_error(f"Timeout analyzing ticker {ticker}")
        return None
    except Exception as e:
        log_error(f"Analyze ticker {ticker} failed: {e}")
        return None

async def analyze_ticker_inner(ticker: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
    try:
        # Historical prices (FMP → fallback yfinance)
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries=400&apikey={FMP_API_KEY}"
        async with session.get(url, timeout=30) as response:
            if response.status == 200:
                data = await response.json()
            else:
                data = {}

        if data.get('historical'):
            df = pd.DataFrame(data['historical'])
            df = df.rename(columns={'date':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
        else:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty:
                log_error(f"No historical data for {ticker}")
                return None
            df = df.rename(columns=str.title)

        if len(df) < 120:
            log_error(f"Insufficient data for {ticker} ({len(df)} rows)")
            return None

        # Fundamentals / profile
        url_info = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
        try:
            async with session.get(url_info, timeout=20) as response:
                info_data = await response.json()
                info = info_data[0] if info_data else {}
        except Exception as e:
            log_error(f"Profile fetch failed for {ticker}: {e}")
            info = {}

        # VIX (yfinance)
        vix_df = None
        for attempt in range(MAX_API_RETRIES):
            try:
                vix_df = yf.download("^VIX", period="2y", interval="1d", progress=False)
                if not vix_df.empty:
                    break
                await asyncio.sleep(RETRY_DELAY)
            except Exception as e:
                log_error(f"VIX fetch error for {ticker} (attempt {attempt+1}): {e}")
                if attempt < MAX_API_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)

        # External signals
        google_trend = await get_google_trends(ticker)
        reddit_sentiment = await analyze_reddit_sentiment(ticker)

        # Features & target
        df_feat = prepare_features(df, info, vix_df)
        if df_feat.empty:
            log_error(f"No features for {ticker}")
            return None
        X, y = timeseries_train_test(df_feat)
        if len(X) < 100:
            log_error(f"Not enough samples after feature engineering for {ticker}")
            return None

        # Classical ensemble with TSCV
        tscv = TimeSeriesSplit(n_splits=5)
        base_model = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=300, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=400, random_state=42, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1))
        ])

        rmses = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            base_model.fit(X_train, y_train)
            y_pred = base_model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            rmses.append(rmse)
        cv_rmse = float(np.mean(rmses))

        # Fit on full data
        base_model.fit(X, y)
        last_features = X.iloc[[-1]]
        voting_pred = float(base_model.predict(last_features)[0])

        # LSTM on Close (univariate) with caching
        models_dir = os.path.join("models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"lstm_{ticker}.keras")

        close_values = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(close_values)

        def make_sequences(arr: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
            Xs, ys = [], []
            for i in range(len(arr) - lookback - 1):
                Xs.append(arr[i:i+lookback])
                ys.append(arr[i+lookback])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = make_sequences(scaled_close, LOOKBACK)
        lstm_pred_price = None
        try:
            if os.path.exists(model_path):
                lstm = load_model(model_path)
            else:
                lstm = build_lstm_model((LOOKBACK, 1))
                callbacks = [
                    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                    ModelCheckpoint(model_path, monitor='loss', save_best_only=True)
                ]
                lstm.fit(X_seq, y_seq, epochs=30, batch_size=32, verbose=0, callbacks=callbacks)
                lstm.save(model_path)
            last_seq = scaled_close[-LOOKBACK:].reshape(1, LOOKBACK, 1)
            lstm_next_scaled = lstm.predict(last_seq, verbose=0)[0][0]
            lstm_pred_price = float(scaler.inverse_transform([[lstm_next_scaled]])[0][0])
        except Exception as e:
            log_error(f"LSTM failed for {ticker}: {e}")

        # Combine predictions
        if lstm_pred_price is None:
            predicted_price = voting_pred
        else:
            predicted_price = 0.8 * voting_pred + 0.2 * lstm_pred_price

        current_price = float(df['Close'].iloc[-1])
        predicted_gain = float((predicted_price - current_price) / current_price)

        # Risk metrics / trade plan
        atr_val = float(atr_series(df).iloc[-1])
        target_price = current_price + atr_val
        stop_loss = current_price - atr_val

        # Kelly (constant params)
        def kelly_criterion(win_prob=0.55, win_loss_ratio=1.8, max_fraction=0.15) -> float:
            f_star = win_prob - (1 - win_prob) / win_loss_ratio
            return float(max(0.0, min(f_star, max_fraction)))
        position_size = kelly_criterion()

        # Feature importances (from RF only for interpretability)
        try:
            rf = base_model.estimators_[0]
            fi = rf.feature_importances_.tolist()
        except Exception:
            fi = []

        out = {
            'ticker': ticker,
            'current_price': current_price,
            'predicted_price': float(predicted_price),
            'predicted_gain': predicted_gain,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'score': float(-cv_rmse),  # lower RMSE → higher score
            'position_size': position_size,
            'google_trend': float(google_trend),
            'reddit_sentiment': float(reddit_sentiment),
            'short_interest': float(info.get('shortPercentOfFloat', 0) or 0),
            'feature_importance': str(fi),
        }
        return out

    except Exception as e:
        shape = 'empty'
        try:
            shape = str(df.shape) if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty else 'empty'
        except Exception:
            pass
        log_error(f"Analyze ticker {ticker} failed inner: {e} - Data shape: {shape}")
        return None

# ============================== #
#            Scanner             #
# ============================== #
async def scan_stocks() -> List[Dict[str, Any]]:
    try:
        results: List[Dict[str, Any]] = []
        tickers = await fetch_tickers()
        if not tickers:
            await send_telegram("⚠️ שגיאה: לא נמצאו טיקרים מתאימים")
            return []

        logging.info(f"Starting analysis of {len(tickers)} tickers")
        tasks = [analyze_ticker(t) for t in tickers]
        analyses = await asyncio.gather(*tasks, return_exceptions=True)

        for i, analysis in enumerate(analyses):
            t = tickers[i]
            ok = (analysis is not None) and (not isinstance(analysis, Exception))
            logging.info(f"Processed {t}: {'Success' if ok else 'Failed'}")
            if not ok:
                log_error(f"Analysis failed for {t}: {analysis}")
                continue
            if analysis['predicted_gain'] > GAIN_THRESHOLD:
                results.append(analysis)

        # Persist & notify
        if results:
            df_results = pd.DataFrame(results)
            # CSV
            df_results.to_csv(OUTPUT_FILE, index=False)

            # DB inserts
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                for row in results:
                    try:
                        cursor.execute(
                            '''INSERT OR IGNORE INTO trades (
                                ticker, entry_price, target_price, stop_loss, score, predicted_gain,
                                days_in_trade, position_size, timestamp, google_trend, reddit_sentiment,
                                short_interest, feature_importance
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (
                                row['ticker'], row['current_price'], row['target_price'], row['stop_loss'],
                                row['score'], row['predicted_gain'], 0, row['position_size'],
                                datetime.utcnow().isoformat(), row['google_trend'], row['reddit_sentiment'],
                                row['short_interest'], row['feature_importance']
                            )
                        )
                    except Exception as e:
                        log_error(f"DB insert failed for {row['ticker']}: {e}")
                conn.commit()

            # Telegram message
            msg_lines = [f"📊 נמצאו {len(results)} מניות מבטיחות:"]
            for row in results:
                msg_lines.append(
                    (f"\nמניה: {row['ticker']}\n"
                     f"מחיר נוכחי: ${row['current_price']:.2f}\n"
                     f"יעד (ATR): ${row['target_price']:.2f}\n"
                     f"סטופ (ATR): ${row['stop_loss']:.2f}\n"
                     f"תחזית: ${row['predicted_price']:.2f} ({row['predicted_gain']*100:.2f}%)\n"
                     f"גודל פוזיציה: {row['position_size']*100:.2f}%")
                )
            await send_telegram("\n".join(msg_lines))
        else:
            await send_telegram("😕 לא נמצאו מניות עם פוטנציאל רווח מעל הסף")
        return results

    except Exception as e:
        log_error(f"Scan stocks error: {e}")
        return []

# ============================== #
#           Entry Point          #
# ============================== #
if __name__ == "__main__":
    init_db()
    asyncio.run(scan_stocks())
