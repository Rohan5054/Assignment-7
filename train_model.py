import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv('train.csv')

# Select features and target
target = 'SalePrice'
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt', 'TotRmsAbvGrd', 'Fireplaces']

# Clean data (drop rows with missing values in features + target)
data = data[features + [target]].dropna()

# Separate features and target
X = data[features]
y = data[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering (optional)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)
X_scaled = np.hstack([X_scaled, data[['Cluster']].values])

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize base models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
ridge_model = Ridge(alpha=1.0)

# Train the RandomForestRegressor model first
rf_model.fit(X_train, y_train)  # Make sure this happens before predictions

# Hyperparameter tuning for Gradient Boosting and Ridge Regression using GridSearch
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2]
}
param_grid_ridge = {
    'alpha': [0.1, 1.0, 10.0]
}

# Perform GridSearchCV
grid_search_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=5, n_jobs=-1, verbose=2)
grid_search_ridge = GridSearchCV(Ridge(), param_grid_ridge, cv=5, n_jobs=-1, verbose=2)

grid_search_gb.fit(X_train, y_train)
grid_search_ridge.fit(X_train, y_train)

best_gb_model = grid_search_gb.best_estimator_
best_ridge_model = grid_search_ridge.best_estimator_

# Train the models
best_gb_model.fit(X_train, y_train)
best_ridge_model.fit(X_train, y_train)

# Create Stacking Model
stacking_model = StackingRegressor(estimators=[
    ('rf', rf_model),
    ('gb', best_gb_model),
    ('ridge', best_ridge_model)
], final_estimator=Ridge())

stacking_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)
gb_pred = best_gb_model.predict(X_test)
ridge_pred = best_ridge_model.predict(X_test)
stacking_pred = stacking_model.predict(X_test)

# Evaluate performance using RMSE
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_pred))

print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Gradient Boosting RMSE: {gb_rmse:.2f}")
print(f"Ridge Regression RMSE: {ridge_rmse:.2f}")
print(f"Stacking RMSE: {stacking_rmse:.2f}")

# Select the best model (lowest RMSE)
best_model = None
if stacking_rmse < rf_rmse and stacking_rmse < gb_rmse and stacking_rmse < ridge_rmse:
    best_model = stacking_model
else:
    best_model = min([(rf_model, rf_rmse), (best_gb_model, gb_rmse), (best_ridge_model, ridge_rmse)],
                     key=lambda x: x[1])[0]

# Define the directory to save models and ensure it exists
model_dir = os.path.join(os.getcwd(), 'model')  # Get current working directory
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Define paths for saving model, scaler, and kmeans
model_path = os.path.join(model_dir, 'best_house_price_model.pkl')
scaler_path = os.path.join(model_dir, 'scaler.pkl')
kmeans_path = os.path.join(model_dir, 'kmeans.pkl')  # Path to save KMeans model

# Save the best model, scaler, and kmeans
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

with open(kmeans_path, 'wb') as f:
    pickle.dump(kmeans, f)

print(f"Best Model saved at {model_path}")
print(f"Scaler saved at {scaler_path}")
print(f"KMeans saved at {kmeans_path}")
