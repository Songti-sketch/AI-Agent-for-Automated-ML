import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime

# Load data
df = pd.read_csv('./resources/data.csv')

# Feature engineering and preprocessing
current_year = datetime.now().year
df['Vehicle_Age'] = current_year - df['Year']
df = df.drop('Description', axis=1)  # Drop unstructured text
df['Location'] = df['Location'].str.strip()
df['Cylinders'] = df['Cylinders'].replace(['nan', 'Unknown'], 'missing')

# Define features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing
numeric_features = ['Year', 'Mileage', 'Vehicle_Age']
categorical_features = ['Make', 'Model', 'Body Type', 'Cylinders', 
                       'Transmission', 'Fuel Type', 'Color', 'Location']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Define base models with optimized hyperparameters
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42
)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    alpha=0.001,
    learning_rate_init=0.01,
    solver='adam',
    max_iter=1000,
    early_stopping=True,
    random_state=42
)

# Create pipelines for each model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', mlp_model)
])

# Train models
rf_pipeline.fit(X_train, y_train)
mlp_pipeline.fit(X_train, y_train)

# Get individual model scores on validation set
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_scores = []
mlp_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    rf_pipeline.fit(X_train_fold, y_train_fold)
    rf_pred = rf_pipeline.predict(X_val_fold)
    rf_scores.append(r2_score(y_val_fold, rf_pred))
    
    mlp_pipeline.fit(X_train_fold, y_train_fold)
    mlp_pred = mlp_pipeline.predict(X_val_fold)
    mlp_scores.append(r2_score(y_val_fold, mlp_pred))

# Calculate weights based on validation performance
rf_weight = np.mean(rf_scores)
mlp_weight = np.mean(mlp_scores)
total_weight = rf_weight + mlp_weight
rf_weight /= total_weight
mlp_weight /= total_weight

print(f"Model weights - Random Forest: {rf_weight:.4f}, MLP: {mlp_weight:.4f}")

# Create ensemble model
ensemble = VotingRegressor(
    estimators=[
        ('rf', rf_pipeline),
        ('mlp', mlp_pipeline)
    ],
    weights=[rf_weight, mlp_weight]
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Evaluate on test set
y_pred = ensemble.predict(X_test)
ensemble_r2 = r2_score(y_test, y_pred)

# Compare with individual models
rf_pred = rf_pipeline.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)

mlp_pred = mlp_pipeline.predict(X_test)
mlp_r2 = r2_score(y_test, mlp_pred)

print(f"\nIndividual Model Performance:")
print(f"Random Forest R2: {rf_r2:.4f}")
print(f"MLP R2: {mlp_r2:.4f}")
print(f"\nEnsemble R2: {ensemble_r2:.4f}")
print(f"Improvement over best individual model: {ensemble_r2 - max(rf_r2, mlp_r2):.4f}")