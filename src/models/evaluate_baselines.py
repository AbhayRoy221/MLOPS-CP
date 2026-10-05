import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score, log_loss, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve

from src.models.train_baselines import PREDICTIVE_FEATURES

def get_metrics(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    roc_auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    ll = log_loss(y, probs)
    
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'log_loss': ll, 'specificity': spec,
        'cm': (tn, fp, fn, tp), 'probs': probs
    }

def run_evaluation():
    os.makedirs('reports/models', exist_ok=True)
    
    splits = {
        'train': pd.read_parquet('data/processed/splits/train.parquet'),
        'validation': pd.read_parquet('data/processed/splits/validation.parquet'),
        'test': pd.read_parquet('data/processed/splits/test.parquet')
    }
    
    results = []
    
    for name in ['lr', 'rf', 'xgb']:
        model_path = f'models/baseline/{name}_pipeline.pkl'
        if not os.path.exists(model_path):
            continue
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        metrics_dict = {'model': name}
        
        for split_name, df in splits.items():
            X = df[PREDICTIVE_FEATURES].copy()
            num_cols = [f for f in PREDICTIVE_FEATURES if f not in ['highest_education', 'code_module', 'code_presentation']]
            for c in num_cols:
                X[c] = pd.to_numeric(X[c], errors='coerce')
                
            y = df['is_withdrawn']
            m = get_metrics(model, X, y)
            
            for k, v in m.items():
                if k not in ['cm', 'probs']:
                    metrics_dict[f'{split_name}_{k}'] = v
            
            if split_name == 'test':
                # Confusion matrix
                tn, fp, fn, tp = m['cm']
                plt.figure()
                plt.matshow([[tn, fp], [fn, tp]], cmap='Blues')
                for (i, j), z in np.ndenumerate([[tn, fp], [fn, tp]]):
                    plt.text(j, i, str(z), ha='center', va='center')
                plt.title(f'Confusion Matrix: {name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.savefig(f'reports/models/{name}_cm.png')
                plt.close()
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y, m['probs'])
                plt.figure()
                plt.plot(fpr, tpr)
                plt.title(f'ROC Curve: {name} (AUC: {m["roc_auc"]:.3f})')
                plt.savefig(f'reports/models/{name}_roc.png')
                plt.close()
                
                # PR Curve
                prec_c, rec_c, _ = precision_recall_curve(y, m['probs'])
                plt.figure()
                plt.plot(rec_c, prec_c)
                plt.title(f'PR Curve: {name} (AUC: {m["pr_auc"]:.3f})')
                plt.savefig(f'reports/models/{name}_pr.png')
                plt.close()
        
        results.append(metrics_dict)
        
        # Feature importance for RF and XGB
        if name in ['rf', 'xgb']:
            clf = model.named_steps['classifier']
            prep = model.named_steps['preprocessor']
            # Get feature names from preprocessor
            num_features = prep.transformers_[0][2]
            num_feature_names = prep.transformers_[0][1].named_steps['imputer'].get_feature_names_out(num_features)
            cat_features = prep.transformers_[1][1].named_steps['encoder'].get_feature_names_out(prep.transformers_[1][2])
            all_features = list(num_feature_names) + list(cat_features)
            
            fi = pd.DataFrame({
                'feature': all_features,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)
            fi.to_csv(f'reports/models/{name}_feature_importance.csv', index=False)
            
    pd.DataFrame(results).to_csv('reports/models/baseline_comparison.csv', index=False)
    print("Evaluation complete.")

if __name__ == "__main__":
    run_evaluation()
