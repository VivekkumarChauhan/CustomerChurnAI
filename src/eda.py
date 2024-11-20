import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_missing_values(data):
    """
    Plot the missing values in the dataset to see which features have missing data.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title("Missing Values in Dataset", fontsize=16)
    plt.show()

def plot_correlation_matrix(data):
    """
    Plot the correlation matrix to see how features are correlated.
    """
    plt.figure(figsize=(12, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, annot_kws={"size": 10})
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()

def plot_distribution(data, column):
    """
    Plot the distribution of a given column to understand its distribution.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

def plot_categorical_distribution(data, column):
    """
    Plot the distribution of a categorical feature.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=data, palette='Set2')
    plt.title(f'Distribution of {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.show()

def plot_boxplot(data, column, target_column):
    """
    Plot a boxplot to see the distribution of a continuous feature for different classes of the target variable.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target_column, y=column, data=data, palette='Set3')
    plt.title(f'Boxplot of {column} by {target_column}', fontsize=16)
    plt.xlabel(target_column, fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.show()

def plot_pairplot(data, target_column):
    """
    Plot a pairplot to examine the relationships between features and the target variable.
    """
    sns.pairplot(data, hue=target_column, plot_kws={'alpha': 0.6, 's': 50})
    plt.suptitle("Pairplot of Features", y=1.02, fontsize=16)
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plot the feature importance based on the trained model.
    Assumes model has an attribute `.feature_importances_` (e.g., RandomForest or XGBoost).
    """
    feature_importance = model.feature_importances_
    sorted_idx = feature_importance.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color='teal')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title("Feature Importance", fontsize=16)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.show()
