from sklearn import ensemble, svm, linear_model, naive_bayes
import xgboost as xgb
import numpy as np

MODELS = {
    # Classification models
    "randomforest_clf": ensemble.RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=7, verbose=2),
    "extratrees_clf": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=7, verbose=2),
    'gradientboost_clf': ensemble.GradientBoostingClassifier(n_estimators=200, random_state=7, verbose=2),
    'xgboost_clf': xgb.XGBClassifier(device='cuda', n_estimators=1000, random_state=7),
    'svm': svm.LinearSVC(random_state=7),
    'logistic_clf': linear_model.LogisticRegression(max_iter=1000, solver='liblinear', random_state=7),
    'nb_clf': naive_bayes.GaussianNB(),

    # Regression models
    'randomforest_reg': ensemble.RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=7, verbose=2),
    'extratrees_reg': ensemble.ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=7, verbose=2),
    'gradientboost_reg': ensemble.GradientBoostingRegressor(n_estimators=200, random_state=7, verbose=2),
    'xgboost_reg': xgb.XGBRegressor(n_estimators=1000, eta=0.001, device='gpu', random_state=7)
}

def get_proba_pred(model, X):
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
            return 1.0 / (1.0 + np.exp(-decisions))
        except (AttributeError, NotImplementedError):
            # For models that don't have decision_function either
            # Just return binary predictions as 0 or 1
            return model.predict(X)
