# DiaLog Deployment Guide (Free Tier Focus)

## Overview
Deploy DiaLog using a free, sustainable stack:
- Frontend (React + Tailwind): Vercel static hosting.
- Backend (FastAPI + Uvicorn): Render free web service.
- Database/Auth/Storage: Firebase (Firestore + Auth + Storage) free tier.
- AI Model: Gemini API key stored as backend environment variable.

This guide covers production hardening, environment variables, build commands, and operational tips.

---
## 1. Prerequisites
- Google Cloud Gemini API key (GEMINI_API_KEY)
- Firebase project (Firestore + Authentication enabled)
- GitHub repository (public or private with proper integration permissions)

Optional:
- Custom domain for frontend (Vercel) and CNAME pointing.

---
## 2. Backend (FastAPI) on Render
### 2.1. Clean Dependencies
Create a production-focused `requirements.prod.txt` (or replace existing duplicates) to reduce cold start:
```
fastapi==0.116.1
uvicorn==0.24.0
pydantic==2.11.7
numpy==2.3.2
pandas==2.3.1
scikit-learn==1.7.1
joblib==1.5.1
python-dotenv==1.1.1
firebase-admin==7.1.0
google-generativeai==0.8.3
google-cloud-firestore==2.21.0
google-cloud-storage==3.3.0
httpx==0.28.1
requests==2.32.4
protobuf==6.32.0
grpcio==1.74.0
grpcio-status==1.74.0
```
(Exclude dev-only or duplicate pins.)

### 2.2. Render Service Setup
1. Log in to Render → New + Web Service.
2. Connect GitHub repo.
3. Root directory: `backend/`
4. Build Command:
```
pip install -r requirements.prod.txt
```
5. Start Command:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```
6. Environment Variables:
   - `GEMINI_API_KEY`= your-key
   - `GEMINI_MODEL`= gemini-2.5-flash (or preferred)
   - `ALLOWED_ORIGINS`= https://your-frontend.vercel.app
   - Optional: `ALLOWED_ORIGIN_REGEX`= https://.*\.vercel\.app
   - Firebase service account JSON: put each field as an env variable OR upload file to Render disk (paid) -> for free tier, prefer storing minimal Firestore credentials via GOOGLE_APPLICATION_CREDENTIALS pattern (requires file; alternative is embed serviceAccount JSON in env and initialize from its string).

### 2.3. CORS
Already configured in `main.py`. Ensure `ALLOWED_ORIGINS` matches deployed frontend.

### 2.4. Health Check
Use `/health` endpoint. Configure Render Health Check Path: `/health`.

### 2.5. Logging & Quotas
Render free tier sleeps after inactivity; first request may be slow. Gemini usage may incur quotas—monitor in Google Cloud console.

---
## 3. Frontend (React) on Vercel
### 3.1. Project Import
1. Log in to Vercel → New Project → Import GitHub repo.
2. Root directory: `frontend/`
3. Framework preset: Create React App.
4. Build Command (auto): `npm run build`
5. Output Directory: `build` (CRA default).

### 3.2. Environment Variables (Vercel Project → Settings → Environment Variables)
- `REACT_APP_API_BASE_URL`= https://your-backend.onrender.com
- (Any additional flags if needed later.)

Redeploy to propagate.

### 3.3. Optional `vercel.json`
Add at repo root (or inside `frontend/` if using Vercel monorepo settings):
```json
{
  "version": 2,
  "builds": [
    { "src": "frontend/package.json", "use": "@vercel/static-build", "config": { "distDir": "build" } }
  ],
  "routes": [
    { "src": "/(.*)", "dest": "/$1" }
  ]
}
```

### 3.4. Cache Busting
CRA handles static asset hashing. Invalidate manually by redeploying.

---
## 4. Firebase Configuration
### 4.1. Firestore Rules (Basic Secure Read/Write for Authenticated Users)
Update `firestore.rules` for least privilege (example):
```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /recommendation_analytics/{docId} {
      allow read, write: if request.auth != null;
    }
  }
}
```
Deploy via Firebase CLI:
```
firebase deploy --only firestore:rules
```

### 4.2. Firestore Indexes
Use `firestore.indexes.json` if compound queries emerge (currently simple adds—optional for now).

### 4.3. Frontend Firebase Config
Keep Firebase web config in the frontend; never expose service account JSON there.

---
## 5. Gemini API
- Store `GEMINI_API_KEY` only on backend.
- Rotate key periodically; avoid exposing it to frontend JavaScript.
- `/ai/env-check` endpoint helps debug prod environment.

---
## 6. Production Hardening
| Concern | Action |
|---------|--------|
| Duplicate deps | Use trimmed `requirements.prod.txt` |
| CORS leakage | Restrict via `ALLOWED_ORIGINS` env |
| Env secret sprawl | Keep secrets only as env vars (Render dashboard) |
| Cold start lag | Warm by periodic ping (e.g., GitHub Action or UptimeRobot) |
| Error visibility | Set `GEMINI_DEBUG_ERRORS=0` in prod |
| Logging volume | Limit Firestore writes to essential events only |

---
## 7. Monitoring & Uptime
- Add UptimeRobot monitor hitting `https://your-backend.onrender.com/health` every 5–10 minutes to keep container warm.
- Track Firestore usage (writes per day). Adjust analytics logging frequency if near limits.

---
## 8. Local Production Simulation
From root:
```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.prod.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd ../frontend
npm install
REACT_APP_API_BASE_URL=http://localhost:8000 npm run build
npx serve -s build
```

---
## 9. Optional Enhancements
- Add GitHub Action to run tests and auto-deploy on tag.
- Introduce Sentry (frontend/backend) for error aggregation (free tier exists).
- Add `/metrics` endpoint with Prometheus format in future if scaling beyond free tier.

---
## 10. Checklist Before Deploy
- [ ] Backend dependencies trimmed.
- [ ] Environment variables set on Render.
- [ ] CORS configured.
- [ ] Frontend built and deployed with correct `REACT_APP_API_BASE_URL`.
- [ ] Firestore rules deployed.
- [ ] Gemini key validated via `/ai/key-quick`.
- [ ] Uptime monitor configured.

---
## 11. Rollback Strategy
- Keep previous successful Render deploy (automatic rollback option exists).
- Vercel keeps deployment history; promote previous build if needed.
- Store service account & Gemini key versions securely (describe in internal ops doc).

---
## 12. Common Issues
| Symptom | Cause | Fix |
|---------|-------|-----|
| 403 on Gemini | Key missing model scope | Regenerate key / verify billing enablement |
| Mixed content errors | Using http backend with https frontend | Change `REACT_APP_API_BASE_URL` to https Render URL |
| CORS blocked | Origin not in ALLOWED_ORIGINS | Update env var and restart service |
| Slow first response | Render free tier cold start | UptimeRobot warm ping |
| Firestore quota warnings | Excess analytics writes | Batch or reduce logging |

---
## 13. Security Notes
- Never ship serviceAccountKey.json publicly; load via env or secure file.
- Restrict Firestore rules early—do not allow public write.
- Validate user inputs server-side (already using Pydantic models).
- Keep dependencies updated quarterly; watch for FastAPI/security CVEs.

---
## 14. When to Upgrade
| Need | Indicator |
|------|-----------|
| Performance | >1s median backend cold starts | Move to paid Render instance |
| Analytics volume | Firestore writes > free quota | Consider BigQuery or consolidated logs |
| Model tuning | Personalized scaling across many users | Introduce background workers / task queue |

---
## 15. Summary
You now have a clear, free-tier deployment pathway: Vercel (frontend) + Render (backend) + Firebase (data/auth) + Gemini (AI). Follow the checklist, apply environment variables carefully, and add monitoring to keep the system reliable.

---
*Last updated: 2025-11-11*
