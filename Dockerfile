FROM python:3.10-slim

WORKDIR /app

# Copy dependency list
COPY requirements-api.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# SHAP 0.49.1 + XGBoost >=2.1 compatibility patch for base_score JSON string parsing
RUN python -c "\
import shap;\
import os;\
tree_file = os.path.join(os.path.dirname(shap.__file__), 'explainers', '_tree.py');\
with open(tree_file, 'r', encoding='utf-8') as f:\
    content = f.read();\
content = content.replace('float(learner_model_param[\"base_score\"])', 'float(str(learner_model_param[\"base_score\"]).strip(\"[]\"))');\
with open(tree_file, 'w', encoding='utf-8') as f:\
    f.write(content);\
"

# Copy API config and source
COPY configs/ configs/
COPY src/api/ src/api/
COPY src/explainability/ src/explainability/
COPY src/intervention/ src/intervention/
COPY src/models/ src/models/

# Copy the model artifact
COPY models/baseline/xgb_pipeline.pkl models/baseline/xgb_pipeline.pkl

# Non-root user setup for security
RUN useradd -m apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Expose port and start FastAPI server
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
