from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats")
def get_stats(request: Request):
    stats = request.app.state.stats
    if stats is None:
        raise HTTPException(status_code=503, detail="stats not available — run the pipeline")
    return stats
