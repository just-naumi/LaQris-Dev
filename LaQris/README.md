# LaQris

> AI-powered object detection web app — **Next.js** frontend · **FastAPI** backend · **YOLOv8** AI

---

## Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14 (TypeScript, Tailwind CSS, App Router) |
| Backend API | FastAPI (Python 3.11+) |
| AI / ML | YOLOv8 via Ultralytics |
| Containerization | Docker + Docker Compose |

---

## Folder Structure

```
LaQris/
├── frontend/          # Next.js app (port 3000)
├── backend/           # FastAPI app (port 8000)
│   ├── app/
│   │   ├── api/v1/    # REST endpoints
│   │   ├── core/      # Config & dependencies
│   │   ├── models/    # Pydantic schemas
│   │   └── services/  # Business logic
│   └── ai/            # YOLO inference layer
│       ├── models/    # Model weights (.pt files)
│       ├── inference.py
│       └── utils.py
└── docker-compose.yml
```

---

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.11+
- (Optional) Docker Desktop

---

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

### Backend (FastAPI + YOLO)

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

> **Note:** YOLOv8n weights (`yolov8n.pt`) will auto-download on first run (~6 MB).

---

### Docker (Both Services)

```bash
docker compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root / welcome |
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/detection/image` | Upload image → YOLO detection results |

---

## Environment Variables

### Backend (`backend/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL_PATH` | `ai/models/yolov8n.pt` | Path to YOLO weights |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Min confidence score |
| `MAX_IMAGE_SIZE_MB` | `10` | Max upload size |

### Frontend (`frontend/.env.local`)
| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | FastAPI base URL (default: `http://localhost:8000`) |
