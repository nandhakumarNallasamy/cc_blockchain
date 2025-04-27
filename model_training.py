#!/usr/bin/env python3
# Train and evaluate machine learning models for credit card fraud detection

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

def load_data(data_dir='./data'):
    """Load the preprocessed data"""
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    # Split features and target
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train, models=None, apply_smote=True):
    """Train multiple models for fraud detection"""
    
    if models is None:
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
    
    # Apply SMOTE to balance the classes if requested
    if apply_smote:
        print("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Original class distribution: {np.bincount(y_train)}")
        print(f"Resampled class distribution: {np.bincount(y_train_resampled)}")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Train models
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_resampled, y_train_resampled, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1'
        )
        
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test, output_dir='./results'):
    """Evaluate models and generate performance metrics"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.png'))
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_roc_curve.png'))
        
        # Generate Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_pr_curve.png'))
        
        # Save feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns
            feature_importance = model.feature_importances_
            
            # Sort features by importance
            sorted_idx = np.argsort(feature_importance)
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.title(f'Feature Importance - {name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{name}_feature_importance.png'))
        
        # Save detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(output_dir, f'{name}_classification_report.csv'))
        
        # Save model
        joblib.dump(model, os.path.join(output_dir, f'{name}_model.pkl'))
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc,
            'confusion_matrix': cm,
            'model_path': os.path.join(output_dir, f'{name}_model.pkl')
        }
    
    # Save results summary
    results_df = pd.DataFrame({
        name: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1'],
            'AUC': metrics['auc']
        }
        for name, metrics in results.items()
    })
    
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    print("\nModel comparison:")
    print(results_df)
    
    return results

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Select best model based on F1 score
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1']:.4f}")
    print(f"Model saved at: {results[best_model_name]['model_path']}")
    
    return best_model

if __name__ == "__main__":
    main()
