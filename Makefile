.PHONY: venv install run docker-build docker-run compose

venv:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip

install:
	. .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && python main.py

docker-build:
	docker build -t penny-bot:latest .

docker-run:
	docker run --rm --name penny-bot --env-file .env \
	  -v $(PWD)/models:/app/models \
	  -v $(PWD)/penny_stocks.db:/app/penny_stocks.db \
	  penny-bot:latest

compose:
	docker compose up --build
