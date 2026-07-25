from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "smart-traffic-api"}

@router.get("/metrics")
async def metrics():
    # Placeholder for Prometheus metrics
    return {"status": "ok"}
