from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api", tags=["blackspots"])


@router.get("/blackspots")
def get_blackspots(request: Request, severe_only: bool = False):
    geo = request.app.state.blackspots
    if geo is None:
        raise HTTPException(status_code=503, detail="blackspots not available — run the pipeline")
    if severe_only:
        feats = [f for f in geo["features"] if f["properties"].get("severe", 0) > 0]
        return {"type": "FeatureCollection", "features": feats}
    return geo
