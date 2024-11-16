# hyperparameter_tuning.py

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

def grid_search_rf(X_train, y_train):
    """
    Perform GridSearchCV for RandomForestClassifier.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters for RandomForest: {grid_search.best_params_}")
    return grid_search.best_estimator_

def grid_search_xgb(X_train, y_train):
    """
    Perform GridSearchCV for XGBClassifier.
    """
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9]
    }
    
    xgb = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters for XGBoost: {grid_search.best_params_}")
    return grid_search.best_estimator_

def randomized_search_svc(X_train, y_train):
    """
    Perform RandomizedSearchCV for SVC.
    """
    param_dist = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    svc = SVC(random_state=42)
    random_search = RandomizedSearchCV(estimator=svc, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=2)
    random_search.fit(X_train, y_train)
    
    print(f"Best Parameters for SVC: {random_search.best_params_}")
    return random_search.best_estimator_
