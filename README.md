# PeopleSense

**Real-time people-counting & crowd-analytics platform built on YOLOv8.**

PeopleSense turns a single-script headcount detector into a full multi-tenant SaaS-style product: REST + WebSocket API, a React dashboard, persistent analytics, per-org auth, configurable alerts, and Docker deployment — all backed by the same YOLOv8 model the original demo used.

> Targets retail footfall analytics, event-crowd monitoring, smart-building occupancy, transit-hub passenger flow, and any other scenario where "how many people are in this frame?" is a question that drives decisions.

---

## Architecture

```
┌────────────────────────────────────┐
│  React + Vite SPA (frontend/)      │
│  Auth · Dashboard · Live View      │
│  Uploads · Analytics · Alerts      │
└─────────────────┬──────────────────┘
                  │  REST + WebSocket
┌─────────────────▼──────────────────┐
│   FastAPI service (backend/)       │
│   JWT auth · OpenAPI docs          │
│   Background job queue             │
└──┬────────────────────────┬────────┘
   │                        │
┌──▼──────────┐    ┌────────▼──────────┐
│ SQLAlchemy  │    │ Detection engine  │
│ SQLite /    │    │ YOLOv8            │
│ Postgres    │    │ + Supervision     │
└─────────────┘    └───────────────────┘
```

| Layer        | Stack                                                          |
| ------------ | -------------------------------------------------------------- |
| Frontend     | React 18, TypeScript, Vite, Tailwind, Zustand, Recharts        |
| Backend      | FastAPI, Pydantic v2, SQLAlchemy 2, Uvicorn                    |
| Auth         | JWT (HS256) + bcrypt, multi-tenant via Organization → Users    |
| Detection    | Ultralytics YOLOv8 (person class) + OpenCV / Pillow            |
| Persistence  | SQLite by default; drop in `DATABASE_URL` for Postgres         |
| Async work   | FastAPI `BackgroundTasks` + DB-tracked `Job` records           |
| Live feed    | WebSocket frame ingestion + JSON detection responses           |
| Alerts       | Threshold rules with optional outbound webhooks                |
| Demo         | Single-file Streamlit app in `streamlit_app/`                  |
| DevOps       | Dockerfile (multi-stage), docker-compose, GitHub Actions CI    |

---

## Repository layout

```
.
├── backend/                  FastAPI service
│   ├── app/
│   │   ├── main.py           FastAPI app factory + middleware
│   │   ├── config.py         pydantic-settings configuration
│   │   ├── db.py             SQLAlchemy engine/session
│   │   ├── deps.py           Auth & DB dependencies
│   │   ├── security.py       JWT + bcrypt helpers
│   │   ├── models/           ORM: User, Org, Camera, Job, DetectionRecord, Alert
│   │   ├── schemas/          Pydantic request/response models
│   │   ├── routers/          /auth /cameras /detect /jobs /analytics /alerts /stream
│   │   ├── services/         Detector, video processor, alert dispatcher
│   │   └── workers/          Background job runners
│   ├── tests/                pytest test-suite (auth, cameras, detect, analytics, alerts)
│   ├── requirements.txt
│   └── pyproject.toml
│
├── frontend/                 React + Vite SPA
│   ├── src/
│   │   ├── api/client.ts     Axios + typed API surface
│   │   ├── auth/store.ts     Persisted Zustand auth store
│   │   ├── components/       Layout, ProtectedRoute
│   │   └── pages/            Login, Register, Dashboard, ImageDetect,
│   │                         VideoDetect, LiveStream, Cameras, Analytics,
│   │                         Alerts, Settings
│   └── tests/                vitest smoke test
│
├── streamlit_app/app.py      Single-file Streamlit demo (cleaned up from
│                             the original three duplicate scripts)
│
├── .github/workflows/ci.yml  Lint + test + build for backend & frontend
├── Dockerfile                Multi-stage: build SPA → install backend → uvicorn
├── docker-compose.yml        backend (always) + frontend (dev profile)
├── Makefile                  install / backend / frontend / test / lint / docker
└── yolov8n.pt, yolov8s.pt    Pretrained YOLOv8 weights (shipped for convenience)
```

---

## Quick start

### Option A — Local dev (two terminals)

```bash
# 1. Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000

# 2. Frontend
cd frontend
npm install
npm run dev
```

Then open:

- Frontend: <http://localhost:5173>
- Swagger docs: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

Register a new account in the UI (this creates a new Organization and a JWT). Every request the SPA makes is authenticated against the same FastAPI service.

### Option B — Docker

```bash
cp .env.example .env
# edit .env, especially SECRET_KEY

docker compose build
docker compose up         # backend only (production-style)
# or with hot-reload frontend:
docker compose --profile dev up
```

- Backend at <http://localhost:8000>
- Frontend dev at <http://localhost:5173>

### Option C — Streamlit demo (the original quick-start)

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

This is the simplified demo that mirrors the original notebook-style scripts (now consolidated into one file).

---

## API surface

All endpoints live under `/api/v1`. Full schema at `/docs`.

| Method  | Path                                | Purpose                                          |
| ------- | ----------------------------------- | ------------------------------------------------ |
| POST    | `/auth/register`                    | Create org + admin user, returns JWT             |
| POST    | `/auth/login`                       | Email/password → JWT                             |
| GET     | `/auth/me`                          | Current user                                     |
| GET/POST/PATCH/DELETE | `/cameras[/{id}]`     | CRUD video sources                               |
| POST    | `/detect/image`                     | Sync: image → count + annotated b64 PNG          |
| POST    | `/detect/video`                     | Async: upload → 202 Job, processed in background |
| GET     | `/detect/status`                    | Detector / model status                          |
| GET     | `/jobs`                             | List recent jobs                                 |
| GET     | `/jobs/{id}`                        | Poll job progress + summary                      |
| GET     | `/jobs/{id}/artifact`               | Download annotated MP4                           |
| GET     | `/analytics/summary?days=7`         | KPIs (peak, average, total)                      |
| GET     | `/analytics/timeseries?days=&bucket_minutes=` | Hourly/daily buckets             |
| GET     | `/analytics/records?limit=`         | Recent detection records                         |
| GET/POST/PATCH/DELETE | `/alerts[/{id}]`      | CRUD threshold alerts                            |
| WS      | `/stream/ws?token=…`                | Stream base64 JPEG frames; receive count + bbox  |

### Example: detect on an image

```bash
TOKEN=$(curl -sS -X POST http://localhost:8000/api/v1/auth/register \
  -H 'content-type: application/json' \
  -d '{"email":"me@example.com","password":"supersecret1","full_name":"Me","organization_name":"Acme"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -sS -X POST http://localhost:8000/api/v1/detect/image \
  -H "authorization: Bearer $TOKEN" \
  -F file=@my-photo.jpg -F confidence=0.4 \
  | jq '{person_count, avg_confidence}'
```

### Example: live WebSocket

The SPA's **Live Stream** page captures `getUserMedia()` frames in a hidden `<canvas>`, posts base64 JPEGs over WebSocket, and renders the annotated PNG returned by the detector at 2–10 FPS.

---

## Configuration

`pydantic-settings` reads from `.env` files (project root or `backend/`) or environment variables.

| Variable                       | Default                                                       | Notes                                                  |
| ------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------ |
| `SECRET_KEY`                   | _insecure default_                                            | **Override in production.** Used to sign JWTs.         |
| `DATABASE_URL`                 | `sqlite:///backend/storage/peoplesense.db`                    | Any SQLAlchemy URL                                     |
| `YOLO_MODEL_PATH`              | `./yolov8n.pt`                                                | Swap for `yolov8s.pt` or a custom-trained model        |
| `YOLO_CONFIDENCE`              | `0.35`                                                        | Default conf threshold                                 |
| `YOLO_IOU`                     | `0.5`                                                         | NMS IoU                                                |
| `ACCESS_TOKEN_EXPIRE_MINUTES`  | `1440`                                                        | JWT lifetime                                           |
| `MAX_UPLOAD_SIZE_MB`           | `200`                                                         | Per-file limit                                         |
| `ENABLE_DETECTOR`              | `true`                                                        | Set `false` in CI / tests to skip loading weights      |
| `CORS_ORIGINS`                 | localhost dev ports                                           | JSON list — set explicitly in prod                     |

See `.env.example` for the complete list.

---

## Testing

```bash
cd backend && pytest -q
cd frontend && npm test
```

- 15 backend pytest cases cover auth, camera CRUD, image detection (with detector mocked off), alerts, analytics summary/timeseries/records.
- Frontend uses `vitest` + `@testing-library/react`; an extensible smoke test starts from `tests/smoke.test.tsx`.

CI runs both suites on every PR (`.github/workflows/ci.yml`).

---

## Production checklist

1. **Set `SECRET_KEY`** to a long random string.
2. **Use Postgres** (`DATABASE_URL=postgresql+psycopg2://…`); enable connection pooling.
3. **Put a reverse proxy** (Caddy/Nginx) in front of Uvicorn for TLS and gzip.
4. **Move the background queue** to Celery + Redis when video volume grows beyond a single worker — the `run_video_job` worker in `backend/app/workers/jobs.py` is intentionally small and stateless so this swap is straightforward.
5. **Mount `backend/storage/`** on persistent disk for video artifacts.
6. **Set `CORS_ORIGINS`** to your actual SPA domain only.
7. **Rotate JWT keys** by changing `SECRET_KEY` (all sessions invalidated).

---

## Roadmap (next obvious steps)

- ByteTrack-based unique-ID tracking + in/out line crossing
- Per-camera heatmap generation
- SSO/OAuth (Google, Azure AD)
- Per-row role-based access control
- Real Celery + Redis worker pool
- Mobile (React Native) live viewer

---

## License

MIT — see [`LICENSE`](LICENSE).
