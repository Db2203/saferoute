import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.graph import load_london_graph
from app.models.routing import (
    ensure_travel_times,
    load_risk_scores_from_db,
    to_routing_scores,
)
from app.models.temporal import load_artifact
from app.routes.hotspots import router as hotspots_router
from app.routes.routing import router as routing_router

log = logging.getLogger("saferoute")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the heavy artifacts once at startup. ~25s cold start (graph build
    # + travel time + risk scores from DB + RF pickle); subsequent requests
    # are cheap because everything lives in app.state.
    log.info("startup: loading graph")
    app.state.graph = ensure_travel_times(load_london_graph())
    log.info("startup: loading risk scores from db")
    app.state.risk_scores = to_routing_scores(load_risk_scores_from_db())
    log.info("startup: loading temporal model")
    app.state.temporal_artifact = load_artifact()
    log.info("startup: ready")
    yield


app = FastAPI(title="SafeRoute API", lifespan=lifespan)

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
app.include_router(routing_router)


@app.get("/health")
def health():
    return {"status": "ok"}
