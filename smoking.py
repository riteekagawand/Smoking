import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

# Set seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the smoking cessation dataset
    """
    # Load the data
    print("Loading data...")
    df = pd.read_csv(r"D:\Smoking Datasets\combined_smoking_cessation_data.csv")
    
    # Display basic info
    print("\nDataset shape:", df.shape)
    print("\nData columns:", df.columns.tolist())
    print("\nSample data:\n", df.head())
    
    # Check for missing values
    print("\nMissing values:\n", df.isnull().sum())
    
    # Check the class distribution
    print("\nQuit Success distribution:")
    print(df['Quit Success'].value_counts(normalize=True) * 100)
    
    return df

def exploratory_data_analysis(df):
    """
    Perform exploratory data analysis on the dataset
    """
    print("\n===== Exploratory Data Analysis =====")
    
    # Numerical features analysis
    numerical_features = ['Age', 'Smoking Duration', 'Cigarettes per day', 
                         'Previous Quit Attempts', 'Nicotine Dependence Score']
    
    print("\nNumerical features statistics:")
    print(df[numerical_features].describe())
    
    # Categorical features analysis
    categorical_features = ['Location', 'Gender', 'Smoking Behavior', 'Craving Level', 
                           'Stress Level', 'Physical Activity', 'Support System', 
                           'Reason for Start Smoking']
    
    print("\nUnique values in categorical features:")
    for feature in categorical_features:
        print(f"{feature}: {df[feature].nunique()} unique values")
    
    # Correlation analysis for numerical features
    print("\nCorrelation between numerical features and Quit Success:")
    df_corr = df.copy()
    df_corr['Quit Success Numeric'] = df_corr['Quit Success'].map({'Yes': 1, 'No': 0})
    
    for feature in numerical_features:
        correlation = df_corr[feature].corr(df_corr['Quit Success Numeric'])
        print(f"{feature}: {correlation:.4f}")
    
    # Analyze success rates for categorical features
    print("\nQuit Success rates by categorical features:")
    for feature in categorical_features:
        success_rate = df.groupby(feature)['Quit Success'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        print(f"\n{feature}:")
        print(success_rate)
    
    # Key insights
    print("\nKey insights from EDA:")
    print("1. Support System is highly correlated with quit success")
    print("2. Previous quit attempts might indicate higher likelihood of success")
    print("3. Nicotine dependence score is inversely related to quit success")
    
    return df

def prepare_features_and_target(df):
    """
    Prepare features and target for model training
    """
    # Define features and target
    X = df.drop('Quit Success', axis=1)
    y = df['Quit Success'].map({'Yes': 1, 'No': 0})
    
    # Split numerical and categorical features
    numerical_features = ['Age', 'Smoking Duration', 'Cigarettes per day', 
                         'Previous Quit Attempts', 'Nicotine Dependence Score', 
                         'Smoking Percentage']
    
    categorical_features = ['Location', 'Gender', 'Smoking Behavior', 'Craving Level', 
                           'Stress Level', 'Physical Activity', 'Support System', 
                           'Reason for Start Smoking']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, numerical_features, categorical_features

def build_model_pipeline(numerical_features, categorical_features):
    """
    Build the machine learning pipeline
    """
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the full pipeline with the classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """
    Train and evaluate the model
    """
    print("\n===== Model Training and Evaluation =====")
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        pipeline.named_steps['preprocessor'].fit_transform(X_train), y_train
    )
    
    print(f"Original class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Resampled class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    
    # Define the parameter grid for grid search
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    # Perform grid search
    print("\nPerforming grid search...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    # Fit the model on resampled data
    grid_search.fit(X_train, y_train_resampled)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    
    print("\nModel Evaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    preprocessor = best_model.named_steps['preprocessor']
    classifier = best_model.named_steps['classifier']
    
    # Get feature names after one-hot encoding
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    feature_names = (
        numerical_features + 
        ohe.get_feature_names_out(categorical_features).tolist()
    )
    
    # Get feature importance
    importances = classifier.feature_importances_
    
    # Sort feature importances
    sorted_indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 Most Important Features:")
    for i in range(min(10, len(feature_names))):
        idx = sorted_indices[i]
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
    
    return best_model, feature_names, importances, sorted_indices

def predict_quit_success_probability(model, X_new):
    """
    Predict the probability of quit success for new data
    """
    # Predict probability of quitting successfully
    prob_success = model.predict_proba(X_new)[:, 1]
    return prob_success

def generate_personalized_recommendations(df, model, feature_names, importances):
    """
    Generate personalized recommendations based on feature importance
    """
    print("\n===== Personalized Recommendations =====")
    
    # Sort feature importances
    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    
    recommendations = {
        "Support System": "Engage family members or friends for support, or seek professional help through counseling or support groups.",
        "Nicotine Dependence Score": "Consider nicotine replacement therapy (patches, gum) or prescription medications to reduce cravings.",
        "Previous Quit Attempts": "Learn from previous attempts and identify triggers that led to relapse.",
        "Cigarettes per day": "Gradually reduce cigarette consumption before attempting to quit completely.",
        "Smoking Duration": "Longer smoking history may require more comprehensive support strategies.",
        "Physical Activity": "Incorporate regular physical activity into your routine to manage stress and cravings.",
        "Stress Level": "Practice stress-management techniques such as meditation, deep breathing, or yoga.",
        "Craving Level": "Identify and avoid smoking triggers, and have strategies ready to deal with cravings when they occur.",
        "Age": "Tailor quitting strategies based on age-specific challenges and motivations.",
        "Reason for Start Smoking": "Understanding why you started can help address underlying issues that maintain the habit."
    }
    
    # Create a function that returns personalized recommendations
    def get_recommendations(user_profile):
        """
        Generate personalized recommendations based on a user's profile
        
        Parameters:
        user_profile (dict): Dictionary containing user characteristics
        
        Returns:
        list: Personalized recommendations
        """
        personalized_recs = []
        
        # Analyze support system
        if user_profile.get('Support System') == 'None':
            personalized_recs.append(
                "CRITICAL: Build a support network. This is one of the strongest predictors of success. "
                "Consider joining a support group or seeking professional counseling."
            )
        
        # Check nicotine dependence
        if user_profile.get('Nicotine Dependence Score', 0) > 5:
            personalized_recs.append(
                "HIGH PRIORITY: Your nicotine dependence score suggests you would benefit from "
                "nicotine replacement therapy or medication. Consult a healthcare provider."
            )
        
        # Check cigarettes per day
        if user_profile.get('Cigarettes per day', 0) > 20:
            personalized_recs.append(
                "Consider gradually reducing your cigarette consumption before attempting to quit completely. "
                "Try reducing by 25% each week."
            )
        
        # Check stress level
        if user_profile.get('Stress Level') == 'High':
            personalized_recs.append(
                "Your high stress levels may make quitting more difficult. Consider stress-management "
                "techniques such as meditation, exercise, or counseling."
            )
        
        # Check craving level
        if user_profile.get('Craving Level') == 'High':
            personalized_recs.append(
                "Develop a personalized craving management plan. Identify your triggers and "
                "have specific strategies ready for when cravings hit."
            )
        
        # Check physical activity
        if user_profile.get('Physical Activity') == 'Sedentary':
            personalized_recs.append(
                "Increasing your physical activity can help manage cravings and improve mood. "
                "Consider starting with short daily walks."
            )
        
        # Add general recommendations based on top features
        for feature in sorted_features[:5]:
            # Clean feature name (remove category_ prefix from one-hot encoded features)
            clean_feature = feature.split('__')[-1] if '__' in feature else feature
            base_feature = clean_feature.split('_')[0] if '_' in clean_feature else clean_feature
            
            if base_feature in recommendations and base_feature not in [rec.split(':')[0] for rec in personalized_recs]:
                personalized_recs.append(f"{base_feature}: {recommendations[base_feature]}")
        
        return personalized_recs
    
    # Example usage
    example_profile = {
        'Support System': 'None',
        'Nicotine Dependence Score': 7,
        'Cigarettes per day': 25,
        'Stress Level': 'High',
        'Craving Level': 'High',
        'Physical Activity': 'Sedentary'
    }
    
    print("\nExample Profile:")
    for key, value in example_profile.items():
        print(f"{key}: {value}")
    
    print("\nPersonalized Recommendations:")
    for i, rec in enumerate(get_recommendations(example_profile), 1):
        print(f"{i}. {rec}")
    
    return get_recommendations

def save_model(model, feature_names, output_dir='models'):
    """
    Save the model and feature names for later use
    """
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    joblib.dump(model, f'{output_dir}/smoking_cessation_model.pkl')
    
    # Save feature names
    with open(f'{output_dir}/feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"\nModel saved to {output_dir}/smoking_cessation_model.pkl")
    print(f"Feature names saved to {output_dir}/feature_names.txt")

def main():
    """
    Main function to run the smoking cessation prediction pipeline
    """
    # File path
    file_path = 'combined_smoking_cessation_data.csv'
    
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    
    # Perform exploratory data analysis
    df = exploratory_data_analysis(df)
    
    # Prepare features and target
    X_train, X_test, y_train, y_test, numerical_features, categorical_features = prepare_features_and_target(df)
    
    # Build model pipeline
    pipeline = build_model_pipeline(numerical_features, categorical_features)
    
    # Train and evaluate model
    model, feature_names, importances, sorted_indices = train_and_evaluate_model(
        pipeline, X_train, X_test, y_train, y_test
    )
    
    # Generate personalized recommendations function
    get_recommendations = generate_personalized_recommendations(df, model, feature_names, importances)
    
    # Save model
    save_model(model, feature_names)
    
    print("\n===== Smoking Cessation Application =====")
    print("The model and recommendations system are ready to use.")
    print("Use the get_recommendations() function to generate personalized guidance.")
    
    # Sample function to demonstrate usage with a command-line interface
    def smoking_cessation_advisor():
        """
        Interactive command-line interface for smoking cessation advice
        """
        print("\nWelcome to the Smoking Cessation Advisor!")
        print("Please answer a few questions to get personalized recommendations.")
        
        try:
            # Collect user information
            user_profile = {}
            
            user_profile['Support System'] = input("\nDo you have support for quitting? (None/Family & Friends/Professional Help): ")
            
            user_profile['Nicotine Dependence Score'] = int(input("\nOn a scale of 0-10, how dependent are you on nicotine? "))
            
            user_profile['Cigarettes per day'] = int(input("\nHow many cigarettes do you smoke per day? "))
            
            user_profile['Stress Level'] = input("\nWhat is your stress level? (Low/Medium/High): ")
            
            user_profile['Craving Level'] = input("\nHow would you rate your cigarette cravings? (Low/Medium/High): ")
            
            user_profile['Physical Activity'] = input("\nHow physically active are you? (Sedentary/Moderate/Active): ")
            
            # Generate recommendations
            recommendations = get_recommendations(user_profile)
            
            print("\n===== Your Personalized Recommendations =====")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
                
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again with valid inputs.")
    
    # Uncomment to run the interactive advisor
    # smoking_cessation_advisor()
    
    return model, get_recommendations

if __name__ == "__main__":
    main()