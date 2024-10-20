import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, PowerTransformer, QuantileTransformer, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np

# Step 2: Preprocess the data
def preprocess_data(df, target_column, scaler_type):
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if there are only numerical or categorical columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # If numerical columns exist, handle scaling
    if len(numerical_cols) > 0:
        # Impute missing values for numerical columns (mean imputation)
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        # Apply scaling based on the selected scaler_type
        scaler = None
        if scaler_type == 'Standard Scaler':
            scaler = StandardScaler()
        elif scaler_type == 'MinMax Scaler':
            scaler = MinMaxScaler()
        elif scaler_type == 'MaxAbs Scaler':
            scaler = MaxAbsScaler()
        elif scaler_type == 'Robust Scaler':
            scaler = RobustScaler()
        elif scaler_type == 'Normalizer':
            scaler = Normalizer()
        elif scaler_type == 'Power Transformer (Yeo-Johnson)':
            scaler = PowerTransformer(method='yeo-johnson')
        elif scaler_type == 'Power Transformer (Box-Cox)':
            scaler = PowerTransformer(method='box-cox')
        elif scaler_type == 'Quantile Transformer':
            scaler = QuantileTransformer(output_distribution='normal')
        elif scaler_type == 'Custom (Log, Exp, etc.)':
            scaler = FunctionTransformer(np.log1p)  # Example using logarithmic transformation

        if scaler is not None:
            X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Handle categorical columns if they exist
    if len(categorical_cols) > 0:
        # Impute missing values for categorical columns (mode imputation)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # One-hot encode categorical features
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test

# Step 3: Train the model
def train_model(X_train, y_train, model, model_name):
    # Train the selected model
    model.fit(X_train, y_train)
    # Save the trained model
    with open(f"{model_name}.pkl", 'wb') as file:
        pickle.dump(model, file)
    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return round(accuracy, 2)
