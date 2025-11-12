# DiaLog – Smart Diabetes Meal Analyzer

DiaLog helps people with diabetes make safer food choices. It analyzes meals using machine learning and provides clear, personalized guidance before and after you eat.

<!-- Badges: Tech, Runtime & License -->
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3-38B2AC?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Firebase](https://img.shields.io/badge/Firebase-Admin-FFCA28?logo=firebase&logoColor=black)](https://firebase.google.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-16%2B-339933?logo=nodedotjs&logoColor=white)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- Features
- Architecture
- Tech Stack
- Getting Started
- Run (Windows PowerShell)
- Multilingual Support 🇮🇳
- Safety & Nutrition Page
- API Overview
- Troubleshooting

## Features

- Smart Meal Analysis (ML) with confidence and explanations
- Truly Personalized Recommendations based on your profile and logs
- Nutritional Facts: calories, carbs, protein, fats, fiber, GI/GL
- Real‑time Search: 1,000+ Indian foods with quick filter
- BMI & Health Insights surfaced in the dashboard
- Risk Assessment Badges: safe / caution / unsafe
- Safety & Nutrition Page to check food safety without logging
- Multilingual UI: popular Indian languages via live translation

## Architecture

- Frontend: React 18, Tailwind CSS, Heroicons
- Backend: FastAPI, Pydantic, Uvicorn, joblib models
- ML: Random Forest/ensemble models + per‑user personalized models
- Data: Food Master Dataset (nutritional facts), User Logs (for personalization)
- Storage/Cloud (optional): Firebase Admin SDK for Firestore logging

See docs for visuals and flows:

- [Architecture Diagram](docs/architecture-diagram.png)
- [ML Workflow](docs/ml-workflow.md)
- [UI Wireframes](docs/ui-wireframes.png)

## Tech Stack

- Frontend: [React](https://react.dev/), [Tailwind CSS](https://tailwindcss.com/), [Heroicons](https://heroicons.com/)
- Backend: [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), [Pydantic](https://docs.pydantic.dev/latest/)
- ML: [scikit‑learn](https://scikit-learn.org/), [joblib](https://joblib.readthedocs.io/)
- Optional: [Firebase Admin](https://firebase.google.com/) for Firestore logging

## Getting Started

Prerequisites:

- Python 3.10+ (3.8+ supported)
- Node.js 16+ (18+ recommended)
- Git

### Initial Setup (first time only)

Backend setup
```powershell
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment (Windows PowerShell)
./venv/Scripts/Activate.ps1

# Install Python dependencies
pip install -r requirements.txt

# Optional: Firebase Admin support (for Firestore logging)
npm install
```

Frontend setup
```powershell
# Navigate to frontend directory
cd ../frontend

# Install Node.js dependencies
npm install
```

## Run (Windows PowerShell)

Option 1: Start both servers together
```powershell
# From project root directory
./start_dev.bat
```

Option 2: Start servers separately
```powershell
# Backend terminal
cd backend; uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend terminal (new window)
cd frontend; npm start
```

Access URLs

- Backend (Production): https://dialog-backend.onrender.com
- Frontend (Production): https://dialog-frontend.onrender.com
- Local Backend API: http://localhost:8000
- Local Frontend Dev: http://localhost:3000 (or 3001 if 3000 is occupied)
- API Docs (Swagger): http://localhost:8000/docs

## Deployment (Render)

### Backend (Web Service)
- workingDirectory: `backend`
- buildCommand: `pip install --upgrade pip ; pip install --only-binary=:all: -r requirements.prod.txt`
- startCommand: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Env vars: `PYTHON_VERSION=3.11.9`, plus any Firebase / Gemini keys.

### Frontend (Static Site)
- Root Directory: `frontend`
- Build Command: `npm ci && npm run build` (or `npm install && npm run build` if no lockfile)
- Publish Directory: `build`
- REACT_APP_API_BASE_URL: `https://dialog-backend.onrender.com`
- Firebase env keys (if using Firestore features).

### SPA Refresh Fix
Static hosting returns 404 on deep links by default. Added `public/404.html` which redirects to `index.html` and a script in `src/index.jsx` that restores the original path for React Router. If you use Render's rewrite settings, also add: Source `/*` → Destination `/index.html` (Rewrite).

### Locking Dependencies (Optional Post‑Deploy)
After a successful deploy you can freeze versions for reproducibility:
```bash
pip freeze > backend/constraints.txt
```
Copy top‑level lines you care about into a pinned `requirements.prod.lock.txt` later. Keep the relaxed file for quick iteration until stability is confirmed.

## Multilingual Support 🇮🇳

- Live translation across the app; choose your language in Profile or via the Navbar’s globe button.
- Popular Indian languages included: Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu (plus English).
- Powered by a free translation API for live content; can be upgraded to i18next + resource files or alternate providers (Bhashini, Sarvam AI, LibreTranslate) as needed.

## Safety & Nutrition Page

Check if a food is Safe / Caution / Unsafe before you eat it – without logging anything.

- UI: Navbar → “Safety & Nutrition” page
- Flow: type to search → pick a food → click “Check Safety”
- Backend: uses `/food/{name}` for nutrition facts and `/predict` for ML safety
- No side effects: nothing is written to Firestore logs from this page

## API Overview

- `GET /health` – server/model status
- `GET /foods` – list of foods
- `GET /food/{food_name}` – nutrition for a food
- `POST /predict` – safety prediction for a meal
- `POST /recommendations` – ML meal recommendations
- `POST /truly-personalized-recommendations` – per‑user model recommendations

See also: [docs/api-endpoints.md](docs/api-endpoints.md)

## Troubleshooting

- Frontend on “port 3000 in use”: it will auto‑select 3001.
- Backend “Could not import module 'main'”: run from `backend` folder or use `uvicorn main:app`.
- Ensure the backend runs on port `8000` to match `frontend/src/services/api.js`.