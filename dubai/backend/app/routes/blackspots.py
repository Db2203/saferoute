from fastapi import APIRouter, HTTPException, Request

from app.data import analytics

router = APIRouter(prefix="/api", tags=["blackspots"])


@router.get("/blackspots")
def get_blackspots(
    request: Request,
    severe_only: bool = False,
    type: str | None = None,
    year: int | None = None,
    hour: int | None = None,
    dow: str | None = None,
    severity: str | None = None,
):
    filters = {
        k: v
        for k, v in {"type": type, "year": year, "hour": hour, "dow": dow, "severity": severity}.items()
        if v is not None
    }
    df = request.app.state.df

    # any cross-filter set -> recompute the grid over that subset (the map
    # follows the dashboard's selection); otherwise serve the precomputed grid.
    if filters and df is not None:
        try:
            return analytics.filtered_grid(df, filters)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    geo = request.app.state.blackspots
    if geo is None:
        raise HTTPException(status_code=503, detail="blackspots not available — run the pipeline")
    if severe_only:
        feats = [f for f in geo["features"] if f["properties"].get("severe", 0) > 0]
        return {"type": "FeatureCollection", "features": feats}
    return geo
