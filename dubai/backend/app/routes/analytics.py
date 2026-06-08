from fastapi import APIRouter, HTTPException, Request

from app.data import analytics

router = APIRouter(prefix="/api", tags=["analytics"])


def _filters(type, year, hour, dow, severity) -> dict:
    f = {"type": type, "year": year, "hour": hour, "dow": dow, "severity": severity}
    return {k: v for k, v in f.items() if v is not None}


@router.get("/analytics")
def get_analytics(
    request: Request,
    type: str | None = None,
    year: int | None = None,
    hour: int | None = None,
    dow: str | None = None,
    severity: str | None = None,
):
    df = request.app.state.df
    if df is None:
        raise HTTPException(status_code=503, detail="analytics not available — run the pipeline")
    try:
        return analytics.compute(df, _filters(type, year, hour, dow, severity))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
