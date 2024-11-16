# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def train_model(data, target_col):
    """
    Train a RandomForest model with hyperparameter tuning and cross-validation.
    Evaluates the model using various metrics and saves the best model.
    """
    # Splitting the data into features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Train-test split (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Mean CV Accuracy: {cv_scores.mean()}')
    
    # Fit the best model on the training data
    best_model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    print(f'ROC AUC Score: {roc_auc:.4f}')
    
    # Feature Importance
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    sorted_idx = feature_importance.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

    # Save the model for future use
    joblib.dump(best_model, 'best_churn_model.pkl')
    
    return best_model, X_test, y_test
