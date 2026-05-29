.PHONY: install backend frontend test lint format docker-build docker-up streamlit

install:
	cd backend && pip install -r requirements-dev.txt
	cd frontend && npm install

backend:
	cd backend && uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

streamlit:
	streamlit run streamlit_app/app.py

test:
	cd backend && pytest -q
	cd frontend && npm test

lint:
	cd backend && ruff check app tests && black --check app tests
	cd frontend && npm run typecheck

format:
	cd backend && ruff check --fix app tests && black app tests

docker-build:
	docker compose build

docker-up:
	docker compose up
