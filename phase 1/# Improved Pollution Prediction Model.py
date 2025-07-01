# Improved Pollution Prediction Model
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load and clean data
df = pd.read_csv('d:/competition/air pollution/phase 1/train.csv')
df_clean = df.dropna()

print(f"Dataset shape after cleaning: {df_clean.shape}")
print(f"Pollution statistics:")
print(f"Mean: {df_clean['pollution_value'].mean():.2f}")
print(f"Median: {df_clean['pollution_value'].median():.2f}")
print(f"Max: {df_clean['pollution_value'].max():.2f}")
print(f"99th percentile: {df_clean['pollution_value'].quantile(0.99):.2f}")

# =====================================
# 1. ADVANCED FEATURE ENGINEERING
# =====================================

def create_enhanced_features(df):
    """Create comprehensive feature set for pollution prediction"""
    df_enhanced = df.copy()
    
    # Cyclical time features (important for temporal patterns)
    df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['day_year_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_year'] / 365)
    df_enhanced['day_year_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_year'] / 365)
    
    # Categorical time features
    df_enhanced['is_weekend'] = (df_enhanced['day_of_week'] >= 5).astype(int)
    df_enhanced['is_rush_hour'] = df_enhanced['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df_enhanced['is_night'] = df_enhanced['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df_enhanced['is_summer'] = df_enhanced['month'].isin([6, 7, 8]).astype(int)
    df_enhanced['is_winter'] = df_enhanced['month'].isin([12, 1, 2]).astype(int)
    
    # Geographic features
    df_enhanced['lat_lon_interaction'] = df_enhanced['latitude'] * df_enhanced['longitude']
    
    # Distance from geographic center
    centroid_lat = df_enhanced['latitude'].mean()
    centroid_lon = df_enhanced['longitude'].mean()
    df_enhanced['distance_from_center'] = np.sqrt(
        (df_enhanced['latitude'] - centroid_lat)**2 + 
        (df_enhanced['longitude'] - centroid_lon)**2
    )
    
    # Geographic clustering
    coords = df_enhanced[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=15, random_state=42)
    df_enhanced['location_cluster'] = kmeans.fit_predict(coords)
    
    # Interaction features
    df_enhanced['hour_month_interaction'] = df_enhanced['hour'] * df_enhanced['month']
    df_enhanced['weekend_hour'] = df_enhanced['is_weekend'] * df_enhanced['hour']
    
    return df_enhanced

# Create enhanced features
df_enhanced = create_enhanced_features(df_clean)

# Define feature columns
base_features = ['latitude', 'longitude', 'day_of_year', 'day_of_week', 'hour', 'month']
cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_year_sin', 'day_year_cos']
categorical_features = ['is_weekend', 'is_rush_hour', 'is_night', 'is_summer', 'is_winter']
geographic_features = ['lat_lon_interaction', 'distance_from_center', 'location_cluster']
interaction_features = ['hour_month_interaction', 'weekend_hour']

all_features = base_features + cyclical_features + categorical_features + geographic_features + interaction_features

print(f"\nTotal features created: {len(all_features)}")
print(f"Feature categories:")
print(f"- Base features: {len(base_features)}")
print(f"- Cyclical features: {len(cyclical_features)}")
print(f"- Categorical features: {len(categorical_features)}")
print(f"- Geographic features: {len(geographic_features)}")
print(f"- Interaction features: {len(interaction_features)}")

# =====================================
# 2. TARGET TRANSFORMATION STRATEGIES
# =====================================

def prepare_targets(y):
    """Prepare different target transformations"""
    targets = {
        'original': y,
        'log': np.log1p(y),
        'sqrt': np.sqrt(y),
        'boxcox': y  # We'll use simple log for now
    }
    
    # Clip extreme outliers (optional)
    y_clipped = np.clip(y, 0, np.percentile(y, 99))
    targets['clipped'] = y_clipped
    targets['log_clipped'] = np.log1p(y_clipped)
    
    return targets

# Prepare target transformations
y_original = df_enhanced['pollution_value']
targets = prepare_targets(y_original)

# =====================================
# 3. PROPER TRAIN/TEST SPLIT
# =====================================

def create_time_based_split(df, test_size=0.2, val_size=0.1):
    """Create time-based split to avoid data leakage"""
    
    # Sort by a time proxy (day_of_year + hour)
    df_sorted = df.sort_values(['day_of_year', 'hour']).reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    train_idx = df_sorted.index[:train_end]
    val_idx = df_sorted.index[train_end:val_end]
    test_idx = df_sorted.index[val_end:]
    
    return train_idx, val_idx, test_idx

# Create time-based split
train_idx, val_idx, test_idx = create_time_based_split(df_enhanced)

print(f"\nDataset split:")
print(f"Training: {len(train_idx)} samples ({len(train_idx)/len(df_enhanced)*100:.1f}%)")
print(f"Validation: {len(val_idx)} samples ({len(val_idx)/len(df_enhanced)*100:.1f}%)")
print(f"Test: {len(test_idx)} samples ({len(test_idx)/len(df_enhanced)*100:.1f}%)")

# =====================================
# 4. MODEL TRAINING AND EVALUATION
# =====================================

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Comprehensive model evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

# Prepare data splits
X = df_enhanced[all_features]
y = df_enhanced['pollution_value']

X_train = X.loc[train_idx]
X_val = X.loc[val_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_val = y.loc[val_idx]
y_test = y.loc[test_idx]

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# =====================================
# 5. MULTIPLE MODEL TRAINING
# =====================================

models = {}
predictions = {}

# 1. LightGBM (typically best for tabular data)
print("\n" + "="*50)
print("Training LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

lgb_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

models['lgb'] = lgb_model
predictions['lgb_val'] = lgb_model.predict(X_val)
predictions['lgb_test'] = lgb_model.predict(X_test)

evaluate_model(y_val, predictions['lgb_val'], "LightGBM")

# 2. Random Forest
print("\n" + "="*50)
print("Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
models['rf'] = rf_model
predictions['rf_val'] = rf_model.predict(X_val_scaled)
predictions['rf_test'] = rf_model.predict(X_test_scaled)

evaluate_model(y_val, predictions['rf_val'], "Random Forest")

# 3. Huber Regressor (robust to outliers)
print("\n" + "="*50)
print("Training Huber Regressor...")
huber_model = HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000)
huber_model.fit(X_train_scaled, y_train)

models['huber'] = huber_model
predictions['huber_val'] = huber_model.predict(X_val_scaled)
predictions['huber_test'] = huber_model.predict(X_test_scaled)

evaluate_model(y_val, predictions['huber_val'], "Huber Regressor")

# =====================================
# 6. ENSEMBLE METHODS
# =====================================

print("\n" + "="*50)
print("Creating Ensemble...")

# Simple averaging ensemble
ensemble_val = (predictions['lgb_val'] + predictions['rf_val'] + predictions['huber_val']) / 3
ensemble_test = (predictions['lgb_test'] + predictions['rf_test'] + predictions['huber_test']) / 3

evaluate_model(y_val, ensemble_val, "Simple Ensemble")

# Weighted ensemble (give more weight to best performing model)
val_scores = {
    'lgb': mean_squared_error(y_val, predictions['lgb_val']),
    'rf': mean_squared_error(y_val, predictions['rf_val']),
    'huber': mean_squared_error(y_val, predictions['huber_val'])
}

# Calculate weights (inverse of MSE, normalized)
weights = {}
total_inv_mse = sum(1/score for score in val_scores.values())
for model, score in val_scores.items():
    weights[model] = (1/score) / total_inv_mse

print(f"\nModel weights:")
for model, weight in weights.items():
    print(f"{model}: {weight:.3f}")

# Weighted ensemble
weighted_ensemble_val = (
    weights['lgb'] * predictions['lgb_val'] + 
    weights['rf'] * predictions['rf_val'] + 
    weights['huber'] * predictions['huber_val']
)

weighted_ensemble_test = (
    weights['lgb'] * predictions['lgb_test'] + 
    weights['rf'] * predictions['rf_test'] + 
    weights['huber'] * predictions['huber_test']
)

evaluate_model(y_val, weighted_ensemble_val, "Weighted Ensemble")

# =====================================
# 7. FEATURE IMPORTANCE ANALYSIS
# =====================================

print("\n" + "="*50)
print("Feature Importance Analysis...")

# LightGBM feature importance
lgb_importance = lgb_model.feature_importance(importance_type='gain')
feature_importance_df = pd.DataFrame({
    'feature': all_features,
    'importance': lgb_importance
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features (LightGBM):")
print(feature_importance_df.head(15).to_string(index=False))

# =====================================
# 8. FINAL TEST EVALUATION
# =====================================

print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

final_models = {
    'LightGBM': predictions['lgb_test'],
    'Random Forest': predictions['rf_test'],
    'Huber Regressor': predictions['huber_test'],
    'Simple Ensemble': ensemble_test,
    'Weighted Ensemble': weighted_ensemble_test
}

test_results = {}
for name, pred in final_models.items():
    test_results[name] = evaluate_model(y_test, pred, name)

# Find best model
best_model = min(test_results.keys(), key=lambda x: test_results[x]['rmse'])
print(f"\nBest performing model: {best_model}")
print(f"Best RMSE: {test_results[best_model]['rmse']:.4f}")

# =====================================
# 9. GENERATE PREDICTIONS FOR SUBMISSION
# =====================================

print("\n" + "="*50)
print("Generating submission predictions...")

# Load test data and apply same preprocessing
test_df = pd.read_csv('d:/competition/air pollution/phase 1/test.csv')
print(f"Test data shape: {test_df.shape}")

# Apply same feature engineering to test data
test_enhanced = create_enhanced_features(test_df)
X_submission = test_enhanced[all_features]
X_submission_scaled = scaler.transform(X_submission)

# Generate predictions using the best ensemble
submission_predictions = (
    weights['lgb'] * lgb_model.predict(X_submission) + 
    weights['rf'] * rf_model.predict(X_submission_scaled) + 
    weights['huber'] * huber_model.predict(X_submission_scaled)
)

# Ensure no negative predictions
submission_predictions = np.maximum(submission_predictions, 0)

# Create submission file
submission_df = pd.DataFrame({
    'id': range(len(submission_predictions)),
    'pollution_value': submission_predictions
})

submission_df.to_csv('d:/competition/air pollution/submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

print(f"\nSubmission statistics:")
print(f"Mean prediction: {submission_predictions.mean():.2f}")
print(f"Median prediction: {np.median(submission_predictions):.2f}")
print(f"Min prediction: {submission_predictions.min():.2f}")
print(f"Max prediction: {submission_predictions.max():.2f}")

# =====================================
# 10. VISUALIZATION AND ANALYSIS
# =====================================

print("\n" + "="*50)
print("Creating visualizations...")

# Create visualizations
plt.figure(figsize=(15, 12))

# 1. Actual vs Predicted scatter plot
plt.subplot(2, 3, 1)
plt.scatter(y_test, weighted_ensemble_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Weighted Ensemble)')

# 2. Residuals plot
plt.subplot(2, 3, 2)
residuals = y_test - weighted_ensemble_test
plt.scatter(weighted_ensemble_test, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

# 3. Feature importance
plt.subplot(2, 3, 3)
top_features = feature_importance_df.head(10)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()

# 4. Model comparison
plt.subplot(2, 3, 4)
model_names = list(test_results.keys())
rmse_scores = [test_results[name]['rmse'] for name in model_names]
plt.bar(model_names, rmse_scores)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)

# 5. Prediction distribution
plt.subplot(2, 3, 5)
plt.hist(submission_predictions, bins=50, alpha=0.7, label='Predictions')
plt.hist(y_train, bins=50, alpha=0.7, label='Training Data')
plt.xlabel('Pollution Value')
plt.ylabel('Frequency')
plt.title('Prediction Distribution vs Training Data')
plt.legend()

# 6. Time series pattern (if applicable)
plt.subplot(2, 3, 6)
# Group by hour and show average pollution
hourly_avg = df_enhanced.groupby('hour')['pollution_value'].mean()
plt.plot(hourly_avg.index, hourly_avg.values, marker='o')
plt.xlabel('Hour of Day')
plt.ylabel('Average Pollution')
plt.title('Average Pollution by Hour')

plt.tight_layout()
plt.savefig('d:/competition/air pollution/model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nModel analysis complete!")
print("="*60)
print(f"Final model performance summary:")
print(f"- Best model: {best_model}")
print(f"- Test RMSE: {test_results[best_model]['rmse']:.4f}")
print(f"- Test R²: {test_results[best_model]['r2']:.4f}")
print(f"- Submission file: submission.csv")
print(f"- Analysis plots: model_analysis.png")
print("="*60)

