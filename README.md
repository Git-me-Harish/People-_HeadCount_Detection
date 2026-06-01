# PeopleSense Cloud

> Multi-tenant crowd intelligence SaaS — real-time head-count detection, predictive analytics, and public transparency for safer public spaces.

---

## Architecture

```
peoplesense/
├── backend/                  # FastAPI + SQLAlchemy
│   ├── app/
│   │   ├── models/           # ORM models (all tenanted by org)
│   │   ├── routers/          # One router per domain
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── services/         # Business logic (detector, notifier, anomaly, heatmap, reporter)
│   │   ├── workers/          # Background job processors
│   │   ├── config.py         # Pydantic settings (env vars)
│   │   ├── db.py             # SQLAlchemy engine + session
│   │   ├── deps.py           # FastAPI dependencies
│   │   ├── main.py           # App factory
│   │   └── security.py       # JWT + bcrypt
│   ├── migrations/           # Alembic migrations
│   └── tests/                # pytest test suite
├── frontend/                 # React + TypeScript + Tailwind
│   ├── src/
│   │   ├── api/              # Typed Axios client
│   │   ├── auth/             # Zustand auth store
│   │   ├── components/       # Layout + reusable UI primitives
│   │   ├── constants/        # All magic values
│   │   ├── hooks/            # Custom React hooks
│   │   ├── i18n/             # English + Hindi translations
│   │   ├── pages/            # Page-level components
│   │   └── types/            # TypeScript domain types
│   └── Dockerfile.frontend
├── docker-compose.yml
└── .env.example
```

---

## Phase 2 Features

| Feature | Status |
|---------|--------|
| Marketing landing page | ✅ |
| 8 industry vertical templates + onboarding wizard | ✅ |
| Multi-camera live grid | ✅ |
| Heatmap density widget | ✅ |
| In-app notifications inbox | ✅ |
| Email / Slack / Teams / Webhook dispatch | ✅ (email logs only — SMTP config required) |
| Customer API tokens (CRUD + one-time reveal) | ✅ |
| Audit log | ✅ |
| PDF reports (ReportLab) | ✅ |
| Public status page | ✅ |
| Z-score anomaly detection | ✅ |
| Plan + usage metering skeleton | ✅ |
| i18n (English + Hindi) | ✅ |
| Dark mode | ✅ |
| Mobile-responsive | ✅ |
| Alembic migrations | ✅ |
| Docker Compose with Postgres + Redis profiles | ✅ |

---

## Quick Start

### Development

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — at minimum set SECRET_KEY

# 2. Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Docker (SQLite)
```bash
docker compose up
```

### Docker (PostgreSQL)
```bash
docker compose --profile postgres up
```

---

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | ⚠️ change-me | JWT signing key |
| `DATABASE_URL` | SQLite | PostgreSQL in prod |
| `YOLO_MODEL_PATH` | `./yolov8n.pt` | YOLOv8 weights |
| `ENABLE_DETECTOR` | `true` | Set `false` in tests |
| `SMTP_HOST` | `""` | SMTP server for email alerts |

---

## API

Swagger UI: `http://localhost:8000/docs`

### Auth flow
```
POST /api/v1/auth/register  → { access_token }
POST /api/v1/auth/login     → { access_token }
Authorization: Bearer <token>
```

### Key endpoints (Phase 2)
```
GET  /api/v1/templates                    # list verticals
POST /api/v1/templates/{vertical}/apply   # apply to org
GET  /api/v1/notifications                # inbox
GET  /api/v1/notifications/count-unread
POST /api/v1/notifications/mark-all-read
GET  /api/v1/api-tokens                   # manage tokens
POST /api/v1/api-tokens
GET  /api/v1/reports/summary/pdf?days=7   # PDF download
GET  /api/v1/public/{slug}                # unauthenticated public view
GET  /api/v1/heatmaps/camera/{id}/latest
GET  /api/v1/plan                         # tier + usage
GET  /api/v1/audit                        # admin only
```

---

## Running Tests

```bash
cd backend
pytest tests/ -v --cov=app

cd frontend
npm test
```

---

## Tier 2 (follow-up)

- Stripe billing checkout
- Twilio SMS alerts
- LSTM-based anomaly forecaster (vs. current z-score)
- AI assistant (NL → SQL)
- Helm chart / Terraform
