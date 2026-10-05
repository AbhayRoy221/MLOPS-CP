import os
import pickle
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.data.split_dataset import build_linear_preprocessor, build_tree_preprocessor
from sklearn.pipeline import Pipeline

PREDICTIVE_FEATURES = [
    'highest_education', 'num_of_prev_attempts', 'studied_credits', 'registration_day',
    'code_module', 'code_presentation', 'module_presentation_length',
    'asm_submission_count', 'asm_mean_score', 'asm_score_std', 'asm_failed_count', 'asm_average_delay',
    'vle_total_clicks', 'vle_active_days', 'vle_days_since_last_activity',
    'vle_clicks_last_7_days', 'vle_clicks_last_14_days', 'vle_forum_clicks',
    'vle_resource_clicks', 'vle_quiz_clicks', 'vle_activity_type_diversity'
]

def load_data():
    train = pd.read_parquet('data/processed/splits/train.parquet')
    X_train = train[PREDICTIVE_FEATURES].copy()
    
    num_cols = [f for f in PREDICTIVE_FEATURES if f not in ['highest_education', 'code_module', 'code_presentation']]
    for c in num_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors='coerce')
        
    y_train = train['is_withdrawn']
    return X_train, y_train

def get_models(config):
    models = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(
        max_iter=config['logistic_regression']['max_iter'],
        random_state=config['random_state']
    )
    num_cols = [f for f in PREDICTIVE_FEATURES if f not in ['highest_education', 'code_module', 'code_presentation']]
    cat_cols = ['highest_education', 'code_module', 'code_presentation']
    
    models['lr'] = Pipeline([
        ('preprocessor', build_linear_preprocessor(num_cols, cat_cols)),
        ('classifier', lr)
    ])
    
    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=config['random_forest']['n_estimators'],
        n_jobs=config['random_forest']['n_jobs'],
        random_state=config['random_state']
    )
    models['rf'] = Pipeline([
        ('preprocessor', build_tree_preprocessor(num_cols, cat_cols)),
        ('classifier', rf)
    ])
    
    # 3. XGBoost
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=config['xgboost']['n_estimators'],
            max_depth=config['xgboost']['max_depth'],
            learning_rate=config['xgboost']['learning_rate'],
            subsample=config['xgboost']['subsample'],
            colsample_bytree=config['xgboost']['colsample_bytree'],
            eval_metric=config['xgboost']['eval_metric'],
            n_jobs=config['xgboost']['n_jobs'],
            random_state=config['random_state'],
            use_label_encoder=False
        )
        models['xgb'] = Pipeline([
            ('preprocessor', build_tree_preprocessor(num_cols, cat_cols)),
            ('classifier', xgb)
        ])
    
    return models

def run_training():
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)['models']
        
    os.makedirs('models/baseline', exist_ok=True)
    
    X_train, y_train = load_data()
    models = get_models(config)
    
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        
        with open(f'models/baseline/{name}_pipeline.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
            
    print("Training complete.")

if __name__ == "__main__":
    run_training()
