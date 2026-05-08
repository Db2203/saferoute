from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.hotspots import router as hotspots_router

app = FastAPI(title="SafeRoute API")

# Allow the local Next.js dev server to call us during development.
# Tighten this once we know where the frontend will actually be hosted.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hotspots_router)


@app.get("/health")
def health():
    return {"status": "ok"}
