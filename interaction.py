import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline  # imblearn's pipeline for SMOTE compatibility
import shap

np.random.seed(42)

def load_and_preprocess_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Handling missing values (Ensuring no data issues)
    df.fillna(method='ffill', inplace=True)

    print("\nDataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nQuit Success distribution:\n", df['Quit Success'].value_counts(normalize=True) * 100)
    return df

def feature_engineering(df):
    print("\n===== Feature Engineering =====")

    # Derived Feature
    df['Smoking Percentage'] = (df['Smoking Duration'] * df['Cigarettes per day']) / df['Age']
    df['Craving-Nicotine Interaction'] = df['Nicotine Dependence Score'] * df['Craving Level'].map({'Low': 1, 'Medium': 2, 'High': 3})

    return df

def prepare_features_and_target(df):
    df = feature_engineering(df)
    
    X = df.drop('Quit Success', axis=1)
    y = df['Quit Success'].map({'Yes': 1, 'No': 0})

    numerical_features = ['Age', 'Smoking Duration', 'Cigarettes per day',
                          'Previous Quit Attempts', 'Nicotine Dependence Score', 
                          'Smoking Percentage', 'Craving-Nicotine Interaction']
    
    categorical_features = ['Location', 'Gender', 'Smoking Behavior',
                            'Craving Level', 'Stress Level', 'Physical Activity',
                            'Support System', 'Reason for Start Smoking']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, numerical_features, categorical_features

def build_model_pipeline(numerical_features, categorical_features):
    num_transformer = SklearnPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = SklearnPipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, numerical_features),
        ('cat', cat_transformer, categorical_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTETomek(random_state=42)),  # Improved Oversampling
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return pipeline

def train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test,
                             numerical_features, categorical_features):
    print("\n===== Training and Evaluation =====")

    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    grid_search = RandomizedSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_iter=10, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("\nBest parameters:", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Feature Importance Analysis
    preprocessor = best_model.named_steps['preprocessor']
    classifier = best_model.named_steps['classifier']
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    feature_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))
    importances = classifier.feature_importances_

    print("\nTop 10 Features:")
    for i in np.argsort(importances)[-10:][::-1]:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

    # SHAP Analysis
    explainer = shap.Explainer(classifier, X_test)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)

    return best_model, feature_names

def save_model(model, feature_names, output_dir='models'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    joblib.dump(model, f'{output_dir}/smoking_model.pkl')
    with open(f'{output_dir}/features.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"\nModel and features saved in '{output_dir}' directory")

def main():
    file_path = r"D:\mini_project\Smoking-Datasets\combined_smoking_cessation_data.csv"
    df = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test, num_feat, cat_feat = prepare_features_and_target(df)
    pipeline = build_model_pipeline(num_feat, cat_feat)
    model, feature_names = train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test,
                                                    num_feat, cat_feat)
    save_model(model, feature_names)

if __name__ == "__main__":
    main()
