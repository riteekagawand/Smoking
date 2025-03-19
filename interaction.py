import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_model(model_path='models/smoking_cessation_model.pkl'):
    """
    Load a saved model from the specified path
    """
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def train_simple_model(file_path='combined_smoking_cessation_data.csv'):
    """
    Train a simple model if a saved model is not available
    """
    print("Training a new model...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Prepare features and target
    X = df.drop('Quit Success', axis=1)
    y = df['Quit Success'].map({'Yes': 1, 'No': 0})
    
    # Split numerical and categorical features
    numerical_features = ['Age', 'Smoking Duration', 'Cigarettes per day', 
                         'Previous Quit Attempts', 'Nicotine Dependence Score', 
                         'Smoking Percentage']
    
    categorical_features = ['Location', 'Gender', 'Smoking Behavior', 'Craving Level', 
                           'Stress Level', 'Physical Activity', 'Support System', 
                           'Reason for Start Smoking']
    
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
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit the model
    model.fit(X, y)
    
    print("Model trained successfully!")
    return model

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
    
    # General recommendations based on common factors
    general_recs = [
        "Set a quit date and prepare for it by removing cigarettes and ashtrays from your environment.",
        "Consider using smoking cessation aids like nicotine patches, gum, or prescription medications.",
        "Identify your smoking triggers and develop strategies to avoid or cope with them.",
        "Practice the 4Ds when cravings hit: Delay, Deep breathe, Drink water, Do something else.",
        "Celebrate your progress and reward yourself for milestones (1 day, 1 week, 1 month smoke-free)."
    ]
    
    # Add general recommendations that don't overlap with personalized ones
    for rec in general_recs:
        if not any(keyword in rec for rec in personalized_recs for keyword in ["support", "nicotine", "cigarette", "stress", "craving", "physical"]):
            personalized_recs.append(rec)
    
    return personalized_recs

def predict_quit_success(model, user_data):
    """
    Predict the probability of quit success for a user
    
    Parameters:
    model: Trained model pipeline
    user_data (dict): Dictionary containing user data
    
    Returns:
    float: Probability of successful quitting
    """
    # Convert user data to DataFrame
    df = pd.DataFrame([user_data])
    
    # Predict probability of quitting successfully
    prob_success = model.predict_proba(df)[:, 1][0]
    
    return prob_success

def smoking_cessation_advisor():
    """
    Interactive command-line interface for smoking cessation advice
    """
    print("\n===== Smoking Cessation Advisor =====")
    print("This tool will help predict your chances of successfully quitting smoking")
    print("and provide personalized recommendations to help you quit.")
    
    # Try to load the model, or train a simple one if loading fails
    try:
        model = load_model()
        if model is None:
            model = train_simple_model()
    except Exception as e:
        print(f"Error: {e}")
        print("Using a simple model instead...")
        model = train_simple_model()
    
    try:
        # Collect user information
        user_profile = {}
        
        # Location (simplified)
        print("\nWhat state do you live in? (Enter state name or 'skip')")
        location = input("> ")
        user_profile['Location'] = location if location.lower() != 'skip' else 'National (States and DC)'
        
        # Gender
        print("\nWhat is your gender? (Male/Female/Overall)")
        gender = input("> ")
        user_profile['Gender'] = gender if gender in ['Male', 'Female'] else 'Overall'
        
        # Smoking behavior
        user_profile['Smoking Behavior'] = 'Cigarette Use (Youth)'
        
        # Age
        print("\nWhat is your age?")
        age = int(input("> "))
        user_profile['Age'] = age
        
        # Smoking duration
        print("\nHow many years have you been smoking?")
        smoking_duration = int(input("> "))
        user_profile['Smoking Duration'] = smoking_duration
        
        # Cigarettes per day
        print("\nHow many cigarettes do you smoke per day?")
        cigarettes_per_day = int(input("> "))
        user_profile['Cigarettes per day'] = cigarettes_per_day
        
        # Previous quit attempts
        print("\nHow many times have you attempted to quit smoking?")
        quit_attempts = int(input("> "))
        user_profile['Previous Quit Attempts'] = quit_attempts
        
        # Craving level
        print("\nHow would you rate your cigarette cravings? (Low/Medium/High)")
        craving_level = input("> ")
        user_profile['Craving Level'] = craving_level
        
        # Stress level
        print("\nHow would you rate your stress level? (Low/Medium/High)")
        stress_level = input("> ")
        user_profile['Stress Level'] = stress_level
        
        # Physical activity
        print("\nHow physically active are you? (Sedentary/Moderate/Active)")
        physical_activity = input("> ")
        user_profile['Physical Activity'] = physical_activity
        
        # Support system
        print("\nWhat kind of support system do you have? (None/Family & Friends/Professional Help)")
        support_system = input("> ")
        user_profile['Support System'] = support_system
        
        # Nicotine dependence score
        print("\nOn a scale of 0-10, how dependent are you on nicotine?")
        dependence_score = int(input("> "))
        user_profile['Nicotine Dependence Score'] = dependence_score
        
        # Reason for starting smoking
        print("\nWhat was your main reason for starting smoking?")
        print("(Curiosity/Peer Pressure/Family Influence/Media Influence/Stress Relief/Other)")
        reason = input("> ")
        user_profile['Reason for Start Smoking'] = reason
        
        # Set a default value for Smoking Percentage (estimated from cigarettes per day)
        # Assuming a population average of about 15% smokers
        user_profile['Smoking Percentage'] = min(cigarettes_per_day * 1.0, 100)
        
        # Generate prediction
        success_probability = predict_quit_success(model, user_profile)
        success_percentage = success_probability * 100
        
        print("\n===== Your Smoking Cessation Analysis =====")
        print(f"Based on your profile, your estimated probability of successfully")
        print(f"quitting smoking is: {success_percentage:.1f}%")
        
        # Interpret the probability
        if success_probability >= 0.7:
            print("\nYou have a good chance of successfully quitting smoking!")
            print("With the right strategies and support, your chances are even better.")
        elif success_probability >= 0.4:
            print("\nYou have a moderate chance of quitting smoking.")
            print("Focus on the recommendations below to significantly improve your odds.")
        else:
            print("\nQuitting may be challenging for you, but it's absolutely possible.")
            print("Many people with similar profiles have successfully quit.")
            print("The recommendations below are especially important for your success.")
        
        # Generate personalized recommendations
        recommendations = get_recommendations(user_profile)
        
        print("\n===== Your Personalized Recommendations =====")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Provide information about resources
        print("\n===== Additional Resources =====")
        print("1. National Quitline: 1-800-QUIT-NOW (1-800-784-8669)")
        print("2. SmokeFree.gov: https://smokefree.gov")
        print("3. American Lung Association: https://www.lung.org/quit-smoking")
        print("4. CDC Tips From Former Smokers: https://www.cdc.gov/tobacco/campaign/tips/")
        
        print("\nRemember, quitting smoking is a journey. Many people make several")
        print("attempts before successfully quitting for good. Each attempt teaches you")
        print("something valuable that can help you succeed next time.")
        
        print("\nWould you like to save your profile and recommendations? (yes/no)")
        save_choice = input("> ")
        
        if save_choice.lower() == "yes":
            filename = f"cessation_plan_{user_profile.get('Location', 'user').replace(' ', '_')}.txt"
            
            with open(filename, 'w') as f:
                f.write("===== SMOKING CESSATION PLAN =====\n\n")
                f.write("=== YOUR PROFILE ===\n")
                for key, value in user_profile.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\n=== YOUR QUIT SUCCESS PROBABILITY ===\n")
                f.write(f"Estimated probability of successfully quitting: {success_percentage:.1f}%\n")
                
                f.write("\n=== YOUR RECOMMENDATIONS ===\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                
                f.write("\n=== RESOURCES ===\n")
                f.write("1. National Quitline: 1-800-QUIT-NOW (1-800-784-8669)\n")
                f.write("2. SmokeFree.gov: https://smokefree.gov\n")
                f.write("3. American Lung Association: https://www.lung.org/quit-smoking\n")
                f.write("4. CDC Tips From Former Smokers: https://www.cdc.gov/tobacco/campaign/tips/\n")
            
            print(f"\nYour cessation plan has been saved to {filename}")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please try again.")

if __name__ == "__main__":
    smoking_cessation_advisor()