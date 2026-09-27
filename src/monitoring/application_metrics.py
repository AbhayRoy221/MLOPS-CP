from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response

# Application metrics
PREDICTION_REQUESTS = Counter(
    'student_dropout_prediction_requests_total',
    'Total number of prediction requests'
)
PREDICTION_ERRORS = Counter(
    'student_dropout_prediction_errors_total',
    'Total number of prediction request errors'
)
PREDICTION_LATENCY = Histogram(
    'student_dropout_prediction_latency_seconds',
    'Latency of prediction requests in seconds'
)

def metrics_endpoint(request: Request):
    """FastAPI route handler for /metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
