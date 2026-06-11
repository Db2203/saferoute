import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import artifacts
from app.routes.analytics import router as analytics_router
from app.routes.blackspots import router as blackspots_router
from app.routes.route import router as route_router
from app.routes.stats import router as stats_router

log = logging.getLogger("saferoute-dubai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load precomputed artifacts once. Missing files -> None (endpoints 503).
    app.state.stats = artifacts.load_stats()
    app.state.blackspots = artifacts.load_blackspots()
    app.state.edge_index = artifacts.load_edge_index()
    app.state.graph = artifacts.load_graph()
    app.state.severity = artifacts.load_severity()
    app.state.df = artifacts.load_collisions()
    log.info(
        "startup: artifacts loaded (graph=%s, collisions=%s)",
        app.state.graph is not None,
        None if app.state.df is None else len(app.state.df),
    )
    yield


app = FastAPI(title="SafeRoute Dubai API", lifespan=lifespan)

# Defaults so endpoints respond gracefully even if lifespan hasn't run (tests).
app.state.stats = None
app.state.blackspots = None
app.state.edge_index = None
app.state.graph = None
app.state.severity = None
app.state.df = None

app.add_middleware(
    CORSMiddleware,
    # Dubai frontend runs on :3001 so it can run alongside the London app (:3000)
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stats_router)
app.include_router(blackspots_router)
app.include_router(route_router)
app.include_router(analytics_router)


@app.get("/health")
def health():
    return {"status": "ok"}
