import streamlit as st
import pandas as pd  # Import pandas
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from ml_utility import (preprocess_data,
                        train_model,
                        evaluate_model)

st.set_page_config(
    page_title="Automate ML",
    page_icon="🧠",
    layout="centered"
)

st.title("🤖 No Code ML Model Training")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    if df is not None:
        st.dataframe(df.head())

        col1, col2, col3, col4 = st.columns(4)

        # Expanded list of scalers with new additions
        scaler_type_list = [
            "None", 
            "Standard Scaler", 
            "MinMax Scaler", 
            "MaxAbs Scaler", 
            "Robust Scaler", 
            "Normalizer",
            "Power Transformer (Yeo-Johnson)", 
            "Power Transformer (Box-Cox)", 
            "Quantile Transformer", 
            "Custom (Log, Exp, etc.)"
        ]

        # Expanded model dictionary with more algorithms
        model_dictionary = {
            "Logistic Regression": LogisticRegression(),
            "Ridge Classifier": RidgeClassifier(),
            "Support Vector Classifier": SVC(),
            "Linear Support Vector Classifier": LinearSVC(max_iter=5000),
            "Random Forest Classifier": RandomForestClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier(),
            "AdaBoost Classifier": AdaBoostClassifier(),
            "Extra Trees Classifier": ExtraTreesClassifier(),
            "XGBoost Classifier": XGBClassifier(),
            "LightGBM Classifier": LGBMClassifier(),
            "CatBoost Classifier": CatBoostClassifier(verbose=0),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "Naive Bayes": GaussianNB(),
            "Neural Network (MLP)": MLPClassifier(max_iter=500)
        }

        with col1:
            target_column = st.selectbox("Select the Target Column", list(df.columns))
        with col2:
            scaler_type = st.selectbox("Select a scaler", scaler_type_list)
        with col3:
            selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
        with col4:
            model_name = st.text_input("Model name")

        if st.button("Train the Model"):

            X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)

            model_to_be_trained = model_dictionary[selected_model]

            model = train_model(X_train, y_train, model_to_be_trained, model_name)

            accuracy = evaluate_model(model, X_test, y_test)

            st.success("Test Accuracy: " + str(accuracy))
