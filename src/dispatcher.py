from sklearn import ensemble, svm, linear_model, naive_bayes
import xgboost as xgb
from sklearn.metrics import roc_auc_score

MODELS = {
    "randomforest_clf": ensemble.RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=7, verbose=2),
    "extratrees_clf": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=7, verbose=2),
    'gradientboost_clf': ensemble.GradientBoostingClassifier(n_estimators=200, random_state=7, verbose=2),
    'xgboost_clf': xgb.XGBClassifier(device='cuda', n_estimators=1000, random_state=7),
    'svm': svm.LinearSVC(random_state=7),
    'logistic_clf': linear_model.LogisticRegression(max_iter=1000, solver='liblinear', random_state=7),
    'nb_clf': naive_bayes.GaussianNB()
}

def get_probability_predictions(model, X):
    """
    Get probability predictions from any model, handling those without predict_proba.
    
    For models with predict_proba: returns probability of class 1
    For models without predict_proba (like LinearSVC): returns normalized decision_function
    """
    try:
        # For models that have predict_proba
        return model.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        # For models that don't have predict_proba but have decision_function
        try:
            # Get decision function scores
            decisions = model.decision_function(X)
            # Scale to approximate probabilities (not calibrated probabilities)
            # Applying a sigmoid-like transformation
            import numpy as np
            return 1.0 / (1.0 + np.exp(-decisions))
        except (AttributeError, NotImplementedError):
            # For models that don't have decision_function either
            # Just return binary predictions as 0 or 1
            return model.predict(X)
