from sklearn import ensemble
import xgboost as xgb
from sklearn.metrics import roc_auc_score

MODELS = {
    "randomforest_clf": ensemble.RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=7, verbose=2),
    "extratrees_clf": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=7, verbose=2),
    'gradientboost_clf': ensemble.GradientBoostingClassifier(n_estimators=200, random_state=7, verbose=2),
    'xgboost_clf': xgb.XGBClassifier(device='cuda', n_estimators=1000, random_state=7)
}
