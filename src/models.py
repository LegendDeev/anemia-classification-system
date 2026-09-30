"""
Model Definitions & Architectural Specifications ("All Five" Algorithms)
=========================================================================
Implements the 5 canonical machine learning classification algorithms:
  1. Logistic Regression (Multinomial with Softmax)
  2. Decision Tree Classifier (CART with Gini Impurity)
  3. Random Forest Classifier (Bagging Ensemble)
  4. Support Vector Machine (RBF Kernel Margin Classifier)
  5. K-Nearest Neighbors (Distance-Weighted Neighborhood Classifier)
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def get_all_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns dictionary containing instances of all 5 classification algorithms
    configured with academically sound hyperparameters.
    """
    models = {
        "logistic_regression": LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
            random_state=random_state
        ),
        "decision_tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=6,
            min_samples_leaf=2,
            criterion="gini",
            random_state=random_state,
            n_jobs=-1
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,  # Enable probability estimates for ROC-AUC
            random_state=random_state
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2  # Euclidean distance
        )
    }
    return models


MODEL_METADATA = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "family": "Generalized Linear Model",
        "mathematical_core": "Multinomial Softmax with Cross-Entropy Loss",
        "key_hyperparameters": "C=1.0 (inverse L2 regularization), solver=lbfgs",
        "strengths": "Fast, interpretable baseline, probabilistic calibrated outputs"
    },
    "decision_tree": {
        "name": "Decision Tree Classifier",
        "family": "Non-parametric Rule-based",
        "mathematical_core": "Recursive Binary Splitting using Gini Impurity",
        "key_hyperparameters": "criterion=gini, max_depth=6, min_samples_split=10",
        "strengths": "White-box explainability, captures non-linear step thresholds"
    },
    "random_forest": {
        "name": "Random Forest Classifier",
        "family": "Bagging Ensemble",
        "mathematical_core": "Bootstrap Aggregation over 100 de-correlated trees",
        "key_hyperparameters": "n_estimators=100, max_depth=8",
        "strengths": "High robustness, resists overfitting, provides feature importance"
    },
    "svm": {
        "name": "Support Vector Machine (SVM)",
        "family": "Maximum-Margin Geometric Classifier",
        "mathematical_core": "Kernel Trick (Radial Basis Function - RBF)",
        "key_hyperparameters": "C=1.0, kernel=rbf, gamma=scale",
        "strengths": "Effective in high-dimensional continuous feature spaces"
    },
    "knn": {
        "name": "K-Nearest Neighbors (KNN)",
        "family": "Instance-based Lazy Learner",
        "mathematical_core": "Inverse-distance weighted Euclidean metric in Voronoi space",
        "key_hyperparameters": "n_neighbors=5, weights=distance, metric=euclidean",
        "strengths": "No explicit training phase assumption, adapts to local cluster geometry"
    }
}
