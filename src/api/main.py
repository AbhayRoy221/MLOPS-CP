from fastapi import FastAPI, Depends, Request, Response, HTTPException
import time
import logging
from src.api.schemas import PredictiveFeatures, PredictionResponse, HealthResponse
from src.api.inference import inference_service
from src.monitoring.application_metrics import PREDICTION_REQUESTS, PREDICTION_ERRORS, PREDICTION_LATENCY, metrics_endpoint

# Setup minimal safe logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Student Dropout Early Warning API",
    version="1.0",
    description="Inference API for predicting student dropout risk and providing interventions."
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    # Minimal safe logging (no full payload, sensitive data, or PII)
    logger.info(f"Method: {request.method} Path: {request.url.path} Status: {response.status_code} Latency: {process_time:.4f}s")
    return response

@app.get("/")
def read_root():
    return {
        "service": "Student Dropout Early Warning API",
        "version": "1.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    loaded = inference_service.is_healthy()
    status = "healthy" if loaded else "unhealthy"
    return HealthResponse(status=status, model_loaded=loaded)

@app.get("/metrics")
def get_metrics(request: Request):
    return metrics_endpoint(request)

@app.post("/predict", response_model=PredictionResponse)
def predict(features: PredictiveFeatures, include_explanation: bool = False):
    PREDICTION_REQUESTS.inc()
    start_time = time.time()
    try:
        result = inference_service.predict(features, include_explanation=include_explanation)
        PREDICTION_LATENCY.observe(time.time() - start_time)
        return result
    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))
