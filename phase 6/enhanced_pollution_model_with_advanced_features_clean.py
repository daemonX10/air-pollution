# =====================================
# ENHANCED POLLUTION PREDICTION MODEL - CORE CODE ONLY
# Focus: LightGBM, XGBoost, Random Forest
# Features: Advanced Feature Engineering, Stacking Ensemble
# =====================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, KFold
from sklearn.preprocessing import RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings
import joblib
import os
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('submissions', exist_ok=True)

print("Enhanced Pollution Prediction Model - Core Code Only")
print("=" * 80)

# =====================================
# ADVANCED FEATURE ENGINEERING
# =====================================

def create_advanced_features_v2(df, is_train=True):
    """
    Advanced feature engineering focusing on spatial, temporal, and interaction features
    """
    df_adv = df.copy()
    
    print(f"Creating advanced features... Initial shape: {df_adv.shape}")
    
    # Preserve target column if it exists (for training data)
    target_col = None
    if 'pollution_value' in df_adv.columns:
        target_col = df_adv['pollution_value'].copy()
        print("Pollution_value target column preserved for feature engineering")
    
    # Handle missing values
    print("Handling missing values...")
    initial_nan_count = df_adv.isnull().sum().sum()
    if initial_nan_count > 0:
        print(f"Found {initial_nan_count} NaN values, filling with appropriate values...")
        
        # Fill numerical columns with median
        numerical_cols = df_adv.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'pollution_value']
        for col in numerical_cols:
            if df_adv[col].isnull().any():
                df_adv[col].fillna(df_adv[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df_adv.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_adv[col].isnull().any():
                df_adv[col].fillna(df_adv[col].mode()[0] if len(df_adv[col].mode()) > 0 else 'unknown', inplace=True)
    
    # 1. POLYNOMIAL INTERACTIONS of key features
    key_features = ['latitude', 'longitude', 'hour']
    available_key_features = [f for f in key_features if f in df_adv.columns]
    
    if len(available_key_features) >= 2:
        print(f"Creating polynomial features from: {available_key_features}")
        
        # Ensure no NaN values in key features
        for feature in available_key_features:
            if df_adv[feature].isnull().any():
                df_adv[feature].fillna(df_adv[feature].median(), inplace=True)
        
        # Check for any infinite values
        for feature in available_key_features:
            df_adv[feature] = df_adv[feature].replace([np.inf, -np.inf], df_adv[feature].median())
        
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(df_adv[available_key_features])
        poly_feature_names = poly.get_feature_names_out(available_key_features)
        
        for i, name in enumerate(poly_feature_names):
            if name not in available_key_features:
                df_adv[f'poly_{name}'] = poly_features[:, i]
        
        print(f"Added {len(poly_feature_names) - len(available_key_features)} polynomial interaction features")
    
    # 2. SPATIAL CLUSTERING
    if 'latitude' in df_adv.columns and 'longitude' in df_adv.columns:
        print("Creating spatial clustering features...")
        coords = df_adv[['latitude', 'longitude']].copy()
        
        # Ensure no NaN values
        coords = coords.fillna(coords.median())
        
        # Create multiple clustering schemes
        for n_clusters in [5, 10, 15]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_adv[f'spatial_cluster_{n_clusters}'] = kmeans.fit_predict(coords)
            
            # Distance to cluster centers
            cluster_centers = kmeans.cluster_centers_
            distances = np.min(np.sqrt(((coords.values[:, np.newaxis] - cluster_centers) ** 2).sum(axis=2)), axis=1)
            df_adv[f'distance_to_cluster_center_{n_clusters}'] = distances
        
        print(f"Added spatial clustering features for 3 different cluster sizes")
    
    # 3. TEMPORAL FEATURES
    if 'hour' in df_adv.columns:
        print("Creating temporal features...")
        
        # Ensure hour is numeric and handle NaN
        df_adv['hour'] = pd.to_numeric(df_adv['hour'], errors='coerce')
        df_adv['hour'] = df_adv['hour'].fillna(df_adv['hour'].median())
        
        # Cyclical encoding for hour
        df_adv['hour_sin'] = np.sin(2 * np.pi * df_adv['hour'] / 24)
        df_adv['hour_cos'] = np.cos(2 * np.pi * df_adv['hour'] / 24)
        
        # Time of day categories
        df_adv['time_of_day'] = pd.cut(df_adv['hour'], 
                                      bins=[0, 6, 12, 18, 24], 
                                      labels=['night', 'morning', 'afternoon', 'evening'],
                                      include_lowest=True)
        
        # Rush hour indicators
        df_adv['is_rush_hour'] = ((df_adv['hour'] >= 7) & (df_adv['hour'] <= 9)) | \
                                ((df_adv['hour'] >= 17) & (df_adv['hour'] <= 19))
        
        print(f"Added temporal features")
    
    # 4. STATISTICAL FEATURES
    print("Creating statistical features...")
    numerical_cols = df_adv.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'pollution_value']
    
    if len(numerical_cols) > 0:
        # Rolling statistics (if we have enough samples)
        if len(df_adv) > 10:
            window_size = min(5, len(df_adv) // 3)
            for col in numerical_cols[:5]:  # Limit to first 5 numeric columns
                if df_adv[col].notna().sum() > window_size:
                    df_adv[f'{col}_rolling_mean'] = df_adv[col].rolling(window=window_size, min_periods=1).mean()
                    df_adv[f'{col}_rolling_std'] = df_adv[col].rolling(window=window_size, min_periods=1).std()
    
    # 5. INTERACTION FEATURES
    if len(numerical_cols) >= 2:
        print("Creating interaction features...")
        # Top numerical features for interactions
        top_features = numerical_cols[:5]  # Limit to avoid explosion
        
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                # Ensure no NaN values
                val1 = df_adv[feat1].fillna(df_adv[feat1].median())
                val2 = df_adv[feat2].fillna(df_adv[feat2].median())
                
                # Basic interactions
                df_adv[f'{feat1}_x_{feat2}'] = val1 * val2
                df_adv[f'{feat1}_plus_{feat2}'] = val1 + val2
                
                # Ratio (avoid division by zero)
                df_adv[f'{feat1}_div_{feat2}'] = val1 / (val2 + 1e-8)
    
    # 6. BINNED FEATURES
    print("Creating binned features...")
    for col in numerical_cols[:3]:  # Limit to first 3 columns
        if df_adv[col].notna().sum() > 10:  # Ensure enough data
            try:
                df_adv[f'{col}_binned'] = pd.qcut(df_adv[col], q=5, labels=False, duplicates='drop')
            except:
                df_adv[f'{col}_binned'] = pd.cut(df_adv[col], bins=5, labels=False)
    
    # 7. CATEGORICAL ENCODING
    print("Encoding categorical features...")
    categorical_cols = df_adv.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col != 'pollution_value':  # Skip target if present
            # One-hot encoding for low cardinality
            unique_values = df_adv[col].nunique()
            if unique_values <= 10:
                dummies = pd.get_dummies(df_adv[col], prefix=col, drop_first=True)
                df_adv = pd.concat([df_adv, dummies], axis=1)
            else:
                # Target encoding for high cardinality (only for training data)
                if is_train and target_col is not None:
                    target_mean = target_col.mean()
                    encoding_map = df_adv.groupby(col)[target_col.name].mean().fillna(target_mean)
                    df_adv[f'{col}_target_encoded'] = df_adv[col].map(encoding_map).fillna(target_mean)
            
            # Remove original categorical column
            df_adv = df_adv.drop(columns=[col])
    
    # Final cleanup
    print("Final cleanup...")
    # Remove any remaining object columns
    object_cols = df_adv.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"Dropping remaining object columns: {list(object_cols)}")
        df_adv = df_adv.drop(columns=object_cols)
    
    # Handle any infinite values
    df_adv = df_adv.replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaN values
    for col in df_adv.columns:
        if col != 'pollution_value' and df_adv[col].isnull().any():
            if df_adv[col].dtype in ['int64', 'float64']:
                df_adv[col] = df_adv[col].fillna(df_adv[col].median())
            else:
                df_adv[col] = df_adv[col].fillna(0)
    
    print(f"Feature engineering completed. Final shape: {df_adv.shape}")
    print(f"Added {df_adv.shape[1] - df.shape[1]} new features")
    
    return df_adv

# =====================================
# DATA LOADING AND PREPROCESSING
# =====================================

print("Loading and preprocessing data...")

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Apply advanced feature engineering
print("\\nApplying feature engineering to training data...")
train_enhanced = create_advanced_features_v2(train_data, is_train=True)

print("\\nApplying feature engineering to test data...")
test_enhanced = create_advanced_features_v2(test_data, is_train=False)

# Align features between train and test
print("\\nAligning features between train and test sets...")
train_features = set(train_enhanced.columns) - {'pollution_value'}
test_features = set(test_enhanced.columns)

# Common features
common_features = train_features.intersection(test_features)
print(f"Common features: {len(common_features)}")

# Keep only common features
X = train_enhanced[list(common_features)].copy()
y = train_enhanced['pollution_value'].copy()
X_test = test_enhanced[list(common_features)].copy()

print(f"Final feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Test matrix shape: {X_test.shape}")

# =====================================
# TARGET TRANSFORMATION
# =====================================

print("\\nApplying target transformation...")

def evaluate_transform(y_orig, transform_name, transformed_y):
    """Evaluate transformation quality"""
    # Normality test
    _, p_value = stats.shapiro(transformed_y[:min(5000, len(transformed_y))])
    
    # Skewness and kurtosis
    skew = stats.skew(transformed_y)
    kurt = stats.kurtosis(transformed_y)
    
    return {
        'transform': transform_name,
        'normality_p': p_value,
        'skewness': abs(skew),
        'kurtosis': abs(kurt),
        'score': p_value + 1/(1 + abs(skew)) + 1/(1 + abs(kurt))
    }

# Test different transformations
transformations = {}

# Original
transformations['original'] = evaluate_transform(y, 'original', y)

# Log transform (only if all values are positive)
if (y > 0).all():
    y_log = np.log1p(y)
    transformations['log'] = evaluate_transform(y, 'log', y_log)

# Square root
y_sqrt = np.sqrt(y - y.min() + 1)
transformations['sqrt'] = evaluate_transform(y, 'sqrt', y_sqrt) 

# Box-Cox (only if all values are positive)
if (y > 0).all():
    try:
        y_boxcox, lambda_boxcox = boxcox(y)
        transformations['boxcox'] = evaluate_transform(y, 'boxcox', y_boxcox)
        transformations['boxcox']['lambda'] = lambda_boxcox
    except:
        print("Box-Cox transformation failed")

# Yeo-Johnson
try:
    y_yeojohnson, lambda_yeojohnson = yeojohnson(y)
    transformations['yeojohnson'] = evaluate_transform(y, 'yeojohnson', y_yeojohnson)
    transformations['yeojohnson']['lambda'] = lambda_yeojohnson
except:
    print("Yeo-Johnson transformation failed")

# Select best transformation
best_transform = max(transformations.keys(), key=lambda k: transformations[k]['score'])
print(f"\\nBest transformation: {best_transform}")

# Apply best transformation
if best_transform == 'log':
    y_transformed = np.log1p(y)
elif best_transform == 'sqrt':
    y_transformed = np.sqrt(y - y.min() + 1)
elif best_transform == 'boxcox':
    y_transformed, _ = boxcox(y)
elif best_transform == 'yeojohnson':
    y_transformed, _ = yeojohnson(y)
else:
    y_transformed = y.copy()

print(f"Target transformation applied: {best_transform}")
print(f"Original target range: [{y.min():.3f}, {y.max():.3f}]")
print(f"Transformed target range: [{y_transformed.min():.3f}, {y_transformed.max():.3f}]")

# =====================================
# ENHANCED CROSS-VALIDATION SETUP
# =====================================

print("\\nSetting up enhanced cross-validation strategy...")

# Split data with stratification based on target quantiles
y_bins = pd.qcut(y_transformed, q=5, labels=False, duplicates='drop')

# Train-validation split with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y_transformed, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_bins
)

print(f"Train set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Enhanced Cross-Validation Strategies
cv_strategies = {
    'KFold': KFold(n_splits=5, shuffle=True, random_state=42),
    'TimeSeriesSplit': TimeSeriesSplit(n_splits=5),
}

print("Cross-validation strategies prepared:")
for name, strategy in cv_strategies.items():
    print(f"  - {name}: {strategy.n_splits} splits")

# Robust Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Data scaling completed using RobustScaler")
print(f"Feature scaling - Train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")

# Feature importance analysis for initial insights
print("\\nComputing feature importance using mutual information...")
feature_importance = mutual_info_regression(X_train_scaled, y_train, random_state=42)
feature_names = X.columns

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"Top 10 most important features:")
print(feature_importance_df.head(10))

print("Enhanced cross-validation setup completed!")

# =====================================
# HYPERPARAMETER OPTIMIZATION
# =====================================

print("\\nStarting hyperparameter optimization...")

def objective_lightgbm(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.LGBMRegressor(**params, n_estimators=500)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='neg_root_mean_squared_error')
    return -cv_scores.mean()

def objective_xgboost(trial):
    params = {
        'n_estimators': 500,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**params)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='neg_root_mean_squared_error')
    return -cv_scores.mean()

def objective_randomforest(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'random_state': 42
    }
    
    model = RandomForestRegressor(**params)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='neg_root_mean_squared_error')
    return -cv_scores.mean()

# Optimize models
models_to_optimize = {
    'lightgbm': objective_lightgbm,
    'xgboost': objective_xgboost,
    'randomforest': objective_randomforest
}

optimized_params = {}
optimized_models = {}

for model_name, objective_func in models_to_optimize.items():
    print(f"\\nOptimizing {model_name.upper()}...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_func, n_trials=50, timeout=300)
    
    optimized_params[model_name] = study.best_params
    
    print(f"Best {model_name} RMSE: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")
    
    # Train final model with best parameters
    if model_name == 'lightgbm':
        optimized_models[model_name] = lgb.LGBMRegressor(**study.best_params, n_estimators=500, random_state=42)
    elif model_name == 'xgboost':
        optimized_models[model_name] = xgb.XGBRegressor(**study.best_params, random_state=42)
    elif model_name == 'randomforest':
        optimized_models[model_name] = RandomForestRegressor(**study.best_params, random_state=42)
    
    optimized_models[model_name].fit(X_train_scaled, y_train)

print(f"\\nHyperparameter optimization completed!")

# =====================================
# MODEL EVALUATION AND ENSEMBLING
# =====================================

print("\\nEvaluating individual models and building ensemble...")

# Evaluate individual models
individual_results = {}
val_predictions = {}
test_predictions = {}

for model_name, model in optimized_models.items():
    # Validation predictions
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    individual_results[model_name] = {
        'rmse': val_rmse,
        'mae': val_mae,
        'r2': val_r2
    }
    
    val_predictions[model_name] = val_pred
    test_predictions[model_name] = test_pred
    
    print(f"{model_name.upper():15} - RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")

# Find best individual model
best_individual_rmse = min(individual_results.values(), key=lambda x: x['rmse'])['rmse']
best_model_name = min(individual_results.keys(), key=lambda k: individual_results[k]['rmse'])

print(f"\\nBest individual model: {best_model_name.upper()} (RMSE: {best_individual_rmse:.6f})")

# Stacking Ensemble
print(f"\\nBuilding stacking ensemble...")

base_models = [
    ('lightgbm', optimized_models['lightgbm']),
    ('xgboost', optimized_models['xgboost']),
    ('randomforest', optimized_models['randomforest'])
]

# Create stacking regressor
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=3
)

stacking_model.fit(X_train_scaled, y_train)
val_pred_stack = stacking_model.predict(X_val_scaled)
test_pred_stack = stacking_model.predict(X_test_scaled)

# Stacking metrics
val_rmse_stack = np.sqrt(mean_squared_error(y_val, val_pred_stack))
val_mae_stack = mean_absolute_error(y_val, val_pred_stack)
val_r2_stack = r2_score(y_val, val_pred_stack)

print(f"STACKING         - RMSE: {val_rmse_stack:.6f}, MAE: {val_mae_stack:.6f}, R²: {val_r2_stack:.6f}")

# Weighted ensemble
print(f"\\nBuilding weighted ensemble...")

# Calculate weights based on inverse RMSE
rmse_values = [individual_results[name]['rmse'] for name in ['lightgbm', 'xgboost', 'randomforest']]
weights = 1 / np.array(rmse_values)
weights = weights / weights.sum()

print(f"Ensemble weights:")
for i, name in enumerate(['lightgbm', 'xgboost', 'randomforest']):
    print(f"  {name}: {weights[i]:.3f}")

# Weighted predictions
weighted_val_pred = sum(weights[i] * val_predictions[name] 
                       for i, name in enumerate(['lightgbm', 'xgboost', 'randomforest']))

weighted_test_pred = sum(weights[i] * test_predictions[name] 
                        for i, name in enumerate(['lightgbm', 'xgboost', 'randomforest']))

weighted_rmse = np.sqrt(mean_squared_error(y_val, weighted_val_pred))
weighted_mae = mean_absolute_error(y_val, weighted_val_pred)
weighted_r2 = r2_score(y_val, weighted_val_pred)

print(f"WEIGHTED         - RMSE: {weighted_rmse:.6f}, MAE: {weighted_mae:.6f}, R²: {weighted_r2:.6f}")

# Select final model
if val_rmse_stack < weighted_rmse and val_rmse_stack < best_individual_rmse:
    final_model = stacking_model
    final_rmse = val_rmse_stack
    final_type = "Stacking"
    final_test_pred = test_pred_stack
elif weighted_rmse < best_individual_rmse:
    final_model = "weighted_ensemble"
    final_rmse = weighted_rmse
    final_type = "Weighted ensemble"
    final_test_pred = weighted_test_pred
else:
    final_model = optimized_models[best_model_name]
    final_rmse = best_individual_rmse
    final_type = f"Individual ({best_model_name})"
    final_test_pred = test_predictions[best_model_name]

print(f"\\n" + "="*50)
print(f"FINAL SELECTED MODEL: {final_type}")
print(f"Final validation RMSE: {final_rmse:.6f}")
print(f"="*50)

# =====================================
# INVERSE TRANSFORMATION AND SUBMISSION
# =====================================

print("\\nApplying inverse transformation and creating submission...")

def inverse_transform_predictions(predictions, transform_method=best_transform):
    """Apply inverse transformation to predictions"""
    if transform_method == 'log':
        return np.expm1(predictions)
    elif transform_method == 'sqrt':
        return (predictions ** 2) + y.min() - 1
    elif transform_method == 'boxcox' and 'lambda' in transformations.get('boxcox', {}):
        lambda_val = transformations['boxcox']['lambda']
        return np.power(predictions * lambda_val + 1, 1 / lambda_val)
    elif transform_method == 'yeojohnson' and 'lambda' in transformations.get('yeojohnson', {}):
        lambda_val = transformations['yeojohnson']['lambda']
        if lambda_val == 0:
            return np.exp(predictions) - 1
        else:
            return np.power(predictions * lambda_val + 1, 1 / lambda_val) - 1
    else:
        return predictions

# Apply inverse transformation
final_test_pred_original = inverse_transform_predictions(final_test_pred)

# Create submission
test_ids = test_data['id'] if 'id' in test_data.columns else range(len(test_data))

submission = pd.DataFrame({
    'id': test_ids,
    'target': final_test_pred_original
})

# Save submission
submission_filename = f'submissions/final_submission_{final_type.lower().replace(" ", "_")}.csv'
submission.to_csv(submission_filename, index=False)

print(f"✅ Final submission saved: {submission_filename}")
print(f"📊 Submission stats:")
print(f"   - Samples: {len(submission)}")
print(f"   - Target range: [{final_test_pred_original.min():.4f}, {final_test_pred_original.max():.4f}]")
print(f"   - Target mean: {final_test_pred_original.mean():.4f}")
print(f"   - Target std: {final_test_pred_original.std():.4f}")

# Save model
if final_model != "weighted_ensemble":
    model_filename = f'models/final_model_{final_type.lower().replace(" ", "_")}.pkl'
    joblib.dump(final_model, model_filename)
    print(f"✅ Final model saved: {model_filename}")

# Save feature names and scaler
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')

print(f"\\n🏁 MODELING PIPELINE COMPLETE!")
print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
