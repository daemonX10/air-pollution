# Comprehensive Multi-Model Air Pollution Prediction with Advanced Features
# =========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                             GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available - skipping CatBoost model")

# Load and clean data
df = pd.read_csv('d:/competition/air pollution/phase 1/train.csv')

# =====================================
# 1. ADVANCED DATA CLEANING
# =====================================

def advanced_data_cleaning(df):
    """Advanced data cleaning with outlier detection"""
    df_clean = df.dropna()
    
    # Detect and handle outliers using IQR method
    Q1 = df_clean['pollution_value'].quantile(0.25)
    Q3 = df_clean['pollution_value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap extreme outliers instead of removing them
    df_clean['pollution_value'] = np.clip(df_clean['pollution_value'], 
                                         lower_bound, 
                                         np.percentile(df_clean['pollution_value'], 99))
    
    print(f"Dataset shape after cleaning: {df_clean.shape}")
    print(f"Outliers capped at: {np.percentile(df_clean['pollution_value'], 99):.2f}")
    
    return df_clean

df_clean = advanced_data_cleaning(df)

# =====================================
# 2. COMPREHENSIVE FEATURE ENGINEERING
# =====================================

def create_advanced_features(df, is_training=True):
    """Create comprehensive feature set without target leakage"""
    df_enhanced = df.copy()
    
    # ===== TEMPORAL FEATURES =====
    
    # Basic cyclical features
    df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['day_year_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_year'] / 365)
    df_enhanced['day_year_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_year'] / 365)
    df_enhanced['day_week_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
    df_enhanced['day_week_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
    
    # Advanced temporal harmonics
    df_enhanced['hour_sin_2nd'] = np.sin(4 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_cos_2nd'] = np.cos(4 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_sin_3rd'] = np.sin(6 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_cos_3rd'] = np.cos(6 * np.pi * df_enhanced['hour'] / 24)
    
    # Monthly harmonics
    df_enhanced['month_sin_2nd'] = np.sin(4 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['month_cos_2nd'] = np.cos(4 * np.pi * df_enhanced['month'] / 12)
    
    # ===== SOLAR AND ATMOSPHERIC PROXIES =====
    
    # Solar elevation angle proxy (based on time and rough latitude)
    day_angle = 2 * np.pi * df_enhanced['day_of_year'] / 365
    declination = 23.45 * np.sin(np.radians(360 * (284 + df_enhanced['day_of_year']) / 365))
    hour_angle = 15 * (df_enhanced['hour'] - 12)
    
    df_enhanced['solar_elevation_proxy'] = np.sin(np.radians(declination)) * np.sin(np.radians(df_enhanced['latitude'])) + \
                                          np.cos(np.radians(declination)) * np.cos(np.radians(df_enhanced['latitude'])) * \
                                          np.cos(np.radians(hour_angle))
    
    # Photoperiod (daylight hours) proxy
    df_enhanced['daylight_proxy'] = 24 - (24/np.pi) * np.arccos(-np.tan(np.radians(df_enhanced['latitude'])) * 
                                                                np.tan(np.radians(declination)))
    
    # UV index proxy
    df_enhanced['uv_proxy'] = np.maximum(0, df_enhanced['solar_elevation_proxy'] * 10)
    
    # ===== ATMOSPHERIC STABILITY PROXIES =====
    
    # Temperature inversion likelihood (based on time and season)
    df_enhanced['inversion_likelihood'] = ((24 - df_enhanced['hour']) % 24) / 24 * \
                                         (1 + 0.5 * np.cos(2 * np.pi * df_enhanced['day_of_year'] / 365))
    
    # Mixing height proxy (higher during day, lower at night)
    df_enhanced['mixing_height_proxy'] = df_enhanced['solar_elevation_proxy'] * \
                                        (1 + 0.3 * np.sin(2 * np.pi * df_enhanced['day_of_year'] / 365))
    
    # Atmospheric stability index
    df_enhanced['stability_index'] = df_enhanced['inversion_likelihood'] / (df_enhanced['mixing_height_proxy'] + 0.1)
    
    # ===== EMISSION SOURCE PROXIES =====
    
    # Traffic emission patterns
    rush_morning = (df_enhanced['hour'] >= 7) & (df_enhanced['hour'] <= 9)
    rush_evening = (df_enhanced['hour'] >= 17) & (df_enhanced['hour'] <= 19)
    weekday = df_enhanced['day_of_week'] < 5
    
    df_enhanced['traffic_intensity'] = (rush_morning.astype(int) * 2 + 
                                       rush_evening.astype(int) * 2 + 
                                       weekday.astype(int)) * \
                                      (1 - 0.5 * (df_enhanced['day_of_week'] >= 5).astype(int))
    
    # Industrial activity proxy
    business_hours = (df_enhanced['hour'] >= 8) & (df_enhanced['hour'] <= 17)
    df_enhanced['industrial_activity'] = business_hours.astype(int) * weekday.astype(int) * \
                                        (1 + 0.2 * np.sin(2 * np.pi * df_enhanced['day_of_year'] / 365))
    
    # Residential heating proxy (seasonal)
    heating_season = (df_enhanced['month'].isin([11, 12, 1, 2, 3])).astype(int)
    evening_morning = ((df_enhanced['hour'] <= 8) | (df_enhanced['hour'] >= 18)).astype(int)
    df_enhanced['heating_emissions'] = heating_season * evening_morning * \
                                      (1 - 0.3 * df_enhanced['solar_elevation_proxy'])
    
    # ===== GEOGRAPHIC AND TOPOGRAPHIC PROXIES =====
    
    # Population density proxy (based on clustering)
    from sklearn.cluster import KMeans
    coords = df_enhanced[['latitude', 'longitude']].values
    
    # Urban centers clustering
    kmeans_urban = KMeans(n_clusters=15, random_state=42, n_init=10)
    urban_clusters = kmeans_urban.fit_predict(coords)
    df_enhanced['urban_cluster'] = urban_clusters
    
    # Distance to urban center
    cluster_centers = kmeans_urban.cluster_centers_
    distances_to_centers = []
    for i, (lat, lon) in enumerate(coords):
        cluster_id = urban_clusters[i]
        center_lat, center_lon = cluster_centers[cluster_id]
        dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
        distances_to_centers.append(dist)
    
    df_enhanced['distance_to_urban_center'] = distances_to_centers
    
    # Population density proxy
    df_enhanced['population_density_proxy'] = 1 / (df_enhanced['distance_to_urban_center'] + 0.001)
    
    # Elevation proxy (rough approximation based on latitude in some regions)
    df_enhanced['elevation_proxy'] = np.abs(df_enhanced['latitude'] - df_enhanced['latitude'].mean()) * 100
    
    # Coastal proximity proxy
    df_enhanced['coastal_proximity'] = np.exp(-np.abs(df_enhanced['longitude'] - df_enhanced['longitude'].mean()) * 10)
    
    # ===== METEOROLOGICAL PROXIES =====
    
    # Wind pattern proxies based on geography and time
    df_enhanced['land_sea_breeze'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24) * df_enhanced['coastal_proximity']
    
    # Mountain valley wind proxy
    df_enhanced['mountain_valley_wind'] = np.sin(2 * np.pi * df_enhanced['hour'] / 12) * df_enhanced['elevation_proxy']
    
    # Pressure gradient proxy (seasonal variation)
    df_enhanced['pressure_gradient_proxy'] = np.sin(2 * np.pi * df_enhanced['day_of_year'] / 365) * \
                                            np.cos(2 * np.pi * df_enhanced['hour'] / 24)
    
    # ===== ADVANCED INTERACTION FEATURES =====
    
    # Emission-meteorology interactions
    df_enhanced['traffic_stability_interaction'] = df_enhanced['traffic_intensity'] * df_enhanced['stability_index']
    df_enhanced['industrial_mixing_interaction'] = df_enhanced['industrial_activity'] * df_enhanced['mixing_height_proxy']
    df_enhanced['heating_inversion_interaction'] = df_enhanced['heating_emissions'] * df_enhanced['inversion_likelihood']
    
    # Geographic-temporal interactions
    df_enhanced['urban_solar_interaction'] = df_enhanced['population_density_proxy'] * df_enhanced['solar_elevation_proxy']
    df_enhanced['coastal_time_interaction'] = df_enhanced['coastal_proximity'] * df_enhanced['hour_sin']
    df_enhanced['elevation_season_interaction'] = df_enhanced['elevation_proxy'] * df_enhanced['month_sin']
    
    # Complex emission combinations
    df_enhanced['total_emission_proxy'] = (df_enhanced['traffic_intensity'] + 
                                          df_enhanced['industrial_activity'] + 
                                          df_enhanced['heating_emissions']) / 3
    
    # ===== FREQUENCY DOMAIN FEATURES =====
    
    # Fourier features for hourly patterns
    df_enhanced['hour_fft_1'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_fft_2'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_fft_3'] = np.sin(4 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_fft_4'] = np.cos(4 * np.pi * df_enhanced['hour'] / 24)
    
    # Weekly patterns
    df_enhanced['week_fft_1'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
    df_enhanced['week_fft_2'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
    
    # ===== STATISTICAL FEATURES (LOCATION-BASED) =====
    
    # Variability measures by location
    df_enhanced['lat_variability'] = np.abs(df_enhanced['latitude'] - df_enhanced['latitude'].median())
    df_enhanced['lon_variability'] = np.abs(df_enhanced['longitude'] - df_enhanced['longitude'].median())
    
    # Extreme location indicators
    df_enhanced['extreme_north'] = (df_enhanced['latitude'] > df_enhanced['latitude'].quantile(0.9)).astype(int)
    df_enhanced['extreme_south'] = (df_enhanced['latitude'] < df_enhanced['latitude'].quantile(0.1)).astype(int)
    df_enhanced['extreme_east'] = (df_enhanced['longitude'] > df_enhanced['longitude'].quantile(0.9)).astype(int)
    df_enhanced['extreme_west'] = (df_enhanced['longitude'] < df_enhanced['longitude'].quantile(0.1)).astype(int)
    
    # ===== POLYNOMIAL AND TRANSFORMATION FEATURES =====
    
    # Polynomial features
    df_enhanced['hour_squared'] = df_enhanced['hour'] ** 2
    df_enhanced['hour_cubed'] = df_enhanced['hour'] ** 3
    df_enhanced['day_year_squared'] = df_enhanced['day_of_year'] ** 2
    df_enhanced['month_squared'] = df_enhanced['month'] ** 2
    
    # Log transformations
    df_enhanced['log_hour_plus1'] = np.log1p(df_enhanced['hour'])
    df_enhanced['log_day_year'] = np.log(df_enhanced['day_of_year'])
    
    # Square root transformations
    df_enhanced['sqrt_hour'] = np.sqrt(df_enhanced['hour'])
    df_enhanced['sqrt_day_year'] = np.sqrt(df_enhanced['day_of_year'])
    
    # ===== RATIO AND PROPORTION FEATURES =====
    
    # Time ratios
    df_enhanced['hour_day_ratio'] = df_enhanced['hour'] / 24
    df_enhanced['day_year_ratio'] = df_enhanced['day_of_year'] / 365
    df_enhanced['month_year_ratio'] = df_enhanced['month'] / 12
    
    # Geographic ratios
    df_enhanced['lat_lon_ratio'] = df_enhanced['latitude'] / (np.abs(df_enhanced['longitude']) + 1e-6)
    df_enhanced['aspect_ratio'] = df_enhanced['latitude'] / (df_enhanced['longitude'] + 1e-6)
    
    # ===== CATEGORICAL ENCODINGS =====
    
    # Time of day categories
    df_enhanced['time_category'] = pd.cut(df_enhanced['hour'], 
                                         bins=[0, 6, 12, 18, 24], 
                                         labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                         include_lowest=True)
    
    # One-hot encode time categories
    time_dummies = pd.get_dummies(df_enhanced['time_category'], prefix='time')
    df_enhanced = pd.concat([df_enhanced, time_dummies], axis=1)
    df_enhanced.drop('time_category', axis=1, inplace=True)
    
    # Season categories
    df_enhanced['season'] = pd.cut(df_enhanced['month'], 
                                  bins=[0, 3, 6, 9, 12], 
                                  labels=['Winter', 'Spring', 'Summer', 'Fall'],
                                  include_lowest=True)
    
    season_dummies = pd.get_dummies(df_enhanced['season'], prefix='season')
    df_enhanced = pd.concat([df_enhanced, season_dummies], axis=1)
    df_enhanced.drop('season', axis=1, inplace=True)
    
    # Fill any NaN values
    df_enhanced = df_enhanced.fillna(0)
    
    return df_enhanced

# =====================================
# 3. MODEL DEFINITIONS
# =====================================

def get_all_models():
    """Define all models to be tested"""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=300, 
            max_depth=20, 
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42, 
            n_jobs=-1
        ),
        
        'Decision Tree': DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),
        
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        
        'SVR': SVR(
            C=100,
            gamma='scale',
            epsilon=0.1
        ),
        
        'KNN': KNeighborsRegressor(
            n_neighbors=10,
            weights='distance',
            n_jobs=-1
        ),
        
        'Linear Regression': LinearRegression(n_jobs=-1),
        
        'Ridge': Ridge(alpha=1.0, random_state=42),
        
        'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
        
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
        
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        
        'AdaBoost': AdaBoostRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostRegressor(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            random_seed=42,
            verbose=False
        )
    
    return models

# =====================================
# 4. MODEL EVALUATION FUNCTIONS
# =====================================

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Comprehensive model evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse
    }

def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """Train and evaluate all models"""
    
    models = get_all_models()
    results = {}
    predictions = {}
    
    print(f"Training {len(models)} models on {X_train.shape[1]} features...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Scale data for models that need it
            if name in ['SVR', 'KNN', 'Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                val_pred = model.predict(X_val_scaled)
                test_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
            
            # Evaluate on validation set
            val_results = evaluate_model(y_val, val_pred, name)
            test_results = evaluate_model(y_test, test_pred, name)
            
            results[name] = {
                'validation': val_results,
                'test': test_results,
                'model': model
            }
            
            predictions[name] = {
                'validation': val_pred,
                'test': test_pred
            }
            
            print(f"Validation RMSE: {val_results['rmse']:.4f}, R²: {val_results['r2']:.4f}")
            print(f"Test RMSE: {test_results['rmse']:.4f}, R²: {test_results['r2']:.4f}")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return results, predictions

# =====================================
# 5. FEATURE SELECTION
# =====================================

def select_best_features(X, y, method='f_regression', k=50):
    """Select best features using various methods"""
    
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
    
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    
    return X_selected, selected_indices, selector

# =====================================
# 6. MAIN EXECUTION
# =====================================

print("Creating comprehensive feature set...")
df_enhanced = create_advanced_features(df_clean, is_training=True)

# Define features (excluding target and id columns)
feature_columns = [col for col in df_enhanced.columns 
                  if col not in ['pollution_value'] and not col.startswith('id')]

print(f"\nTotal features created: {len(feature_columns)}")

# Prepare data
X = df_enhanced[feature_columns].fillna(0)
y = df_enhanced['pollution_value']

print(f"Dataset shape: {X.shape}")
print(f"Feature types: {X.dtypes.value_counts().to_dict()}")

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42
)

print(f"\nDataset split:")
print(f"Training: {len(X_train)} samples")
print(f"Validation: {len(X_val)} samples")
print(f"Test: {len(X_test)} samples")

# Feature selection
print("\nPerforming feature selection...")
X_train_selected, selected_indices, selector = select_best_features(
    X_train, y_train, method='f_regression', k=60
)

X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

selected_features = [feature_columns[i] for i in selected_indices]
print(f"Selected {len(selected_features)} features out of {len(feature_columns)}")

# Train and evaluate all models
print("\n" + "="*80)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*80)

results, predictions = train_and_evaluate_models(
    X_train_selected, X_val_selected, X_test_selected,
    y_train, y_val, y_test, selected_features
)

# =====================================
# 7. RESULTS ANALYSIS
# =====================================

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

# Create results DataFrame
results_data = []
for name, result in results.items():
    results_data.append({
        'Model': name,
        'Val_RMSE': result['validation']['rmse'],
        'Val_R2': result['validation']['r2'],
        'Test_RMSE': result['test']['rmse'],
        'Test_R2': result['test']['r2']
    })

results_df = pd.DataFrame(results_data)
results_df = results_df.sort_values('Test_RMSE')

print("\nModel Performance Ranking (by Test RMSE):")
print("="*60)
for i, row in results_df.iterrows():
    print(f"{row['Model']:20s} | Test RMSE: {row['Test_RMSE']:7.4f} | Test R²: {row['Test_R2']:6.4f}")

print(f"\nBest Model: {results_df.iloc[0]['Model']}")
print(f"Best Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}")
print(f"Best Test R²: {results_df.iloc[0]['Test_R2']:.4f}")

# =====================================
# 8. ENSEMBLE CREATION
# =====================================

print("\n" + "="*50)
print("CREATING ENSEMBLE MODELS")
print("="*50)

# Top 5 models ensemble
top_5_models = results_df.head(5)['Model'].tolist()
print(f"Top 5 models for ensemble: {top_5_models}")

# Simple average ensemble
ensemble_val_pred = np.mean([predictions[model]['validation'] for model in top_5_models], axis=0)
ensemble_test_pred = np.mean([predictions[model]['test'] for model in top_5_models], axis=0)

ensemble_val_results = evaluate_model(y_val, ensemble_val_pred, "Top-5 Ensemble")
ensemble_test_results = evaluate_model(y_test, ensemble_test_pred, "Top-5 Ensemble")

print(f"Ensemble Validation RMSE: {ensemble_val_results['rmse']:.4f}")
print(f"Ensemble Test RMSE: {ensemble_test_results['rmse']:.4f}")

# Weighted ensemble based on inverse RMSE
weights = {}
total_inv_rmse = 0
for model in top_5_models:
    inv_rmse = 1 / results[model]['validation']['rmse']
    weights[model] = inv_rmse
    total_inv_rmse += inv_rmse

# Normalize weights
for model in weights:
    weights[model] /= total_inv_rmse

print(f"\nWeighted ensemble weights:")
for model, weight in weights.items():
    print(f"{model}: {weight:.3f}")

# Weighted ensemble predictions
weighted_val_pred = sum(weights[model] * predictions[model]['validation'] for model in top_5_models)
weighted_test_pred = sum(weights[model] * predictions[model]['test'] for model in top_5_models)

weighted_val_results = evaluate_model(y_val, weighted_val_pred, "Weighted Ensemble")
weighted_test_results = evaluate_model(y_test, weighted_test_pred, "Weighted Ensemble")

print(f"Weighted Ensemble Validation RMSE: {weighted_val_results['rmse']:.4f}")
print(f"Weighted Ensemble Test RMSE: {weighted_test_results['rmse']:.4f}")

# =====================================
# 9. VISUALIZATION
# =====================================

print("\nCreating comprehensive visualizations...")

plt.figure(figsize=(20, 15))

# 1. Model comparison
plt.subplot(3, 4, 1)
plt.barh(range(len(results_df)), results_df['Test_RMSE'])
plt.yticks(range(len(results_df)), results_df['Model'])
plt.xlabel('Test RMSE')
plt.title('Model Performance Comparison')
plt.gca().invert_yaxis()

# 2. R² comparison
plt.subplot(3, 4, 2)
plt.barh(range(len(results_df)), results_df['Test_R2'])
plt.yticks(range(len(results_df)), results_df['Model'])
plt.xlabel('Test R²')
plt.title('R² Score Comparison')
plt.gca().invert_yaxis()

# 3. Validation vs Test performance
plt.subplot(3, 4, 3)
plt.scatter(results_df['Val_RMSE'], results_df['Test_RMSE'])
for i, model in enumerate(results_df['Model']):
    plt.annotate(model, (results_df.iloc[i]['Val_RMSE'], results_df.iloc[i]['Test_RMSE']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.xlabel('Validation RMSE')
plt.ylabel('Test RMSE')
plt.title('Validation vs Test Performance')
plt.plot([0, max(results_df['Val_RMSE'])], [0, max(results_df['Test_RMSE'])], 'r--', alpha=0.5)

# 4. Feature importance (from best tree-based model)
plt.subplot(3, 4, 4)
best_model_name = results_df.iloc[0]['Model']
if best_model_name in ['Random Forest', 'Extra Trees', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
    if hasattr(results[best_model_name]['model'], 'feature_importances_'):
        importance = results[best_model_name]['model'].feature_importances_
        top_indices = np.argsort(importance)[-15:]
        top_features = [selected_features[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title(f'Top 15 Features ({best_model_name})')
        plt.gca().invert_yaxis()

# 5. Residuals plot for best model
plt.subplot(3, 4, 5)
best_pred = predictions[best_model_name]['test']
residuals = y_test - best_pred
plt.scatter(best_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title(f'Residuals Plot ({best_model_name})')

# 6. Actual vs Predicted
plt.subplot(3, 4, 6)
plt.scatter(y_test, best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted ({best_model_name})')

# 7. Model type analysis
plt.subplot(3, 4, 7)
model_types = {
    'Tree-based': ['Random Forest', 'Decision Tree', 'Extra Trees', 'XGBoost', 
                   'LightGBM', 'CatBoost', 'Gradient Boosting', 'AdaBoost'],
    'Linear': ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet'],
    'Instance-based': ['KNN'],
    'Kernel': ['SVR']
}

type_performance = {}
for type_name, models in model_types.items():
    rmse_scores = [results_df[results_df['Model'] == model]['Test_RMSE'].iloc[0] 
                   for model in models if model in results_df['Model'].values]
    if rmse_scores:
        type_performance[type_name] = np.mean(rmse_scores)

if type_performance:
    plt.bar(type_performance.keys(), type_performance.values())
    plt.ylabel('Average Test RMSE')
    plt.title('Performance by Model Type')
    plt.xticks(rotation=45)

# 8. Learning curves for top models
plt.subplot(3, 4, 8)
train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
for model_name in top_5_models[:3]:  # Top 3 models
    rmse_scores = []
    for size in train_sizes:
        n_samples = int(len(X_train_selected) * size)
        X_subset = X_train_selected[:n_samples]
        y_subset = y_train.iloc[:n_samples]
        
        model = results[model_name]['model']
        if model_name in ['SVR', 'KNN', 'Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
            scaler = StandardScaler()
            X_subset_scaled = scaler.fit_transform(X_subset)
            X_val_scaled = scaler.transform(X_val_selected)
            model.fit(X_subset_scaled, y_subset)
            pred = model.predict(X_val_scaled)
        else:
            model.fit(X_subset, y_subset)
            pred = model.predict(X_val_selected)
        
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        rmse_scores.append(rmse)
    
    plt.plot(train_sizes, rmse_scores, marker='o', label=model_name)

plt.xlabel('Training Size Fraction')
plt.ylabel('Validation RMSE')
plt.title('Learning Curves')
plt.legend()

# 9. Ensemble comparison
plt.subplot(3, 4, 9)
ensemble_results = {
    'Best Single': results_df.iloc[0]['Test_RMSE'],
    'Top-5 Average': ensemble_test_results['rmse'],
    'Weighted Ensemble': weighted_test_results['rmse']
}

plt.bar(ensemble_results.keys(), ensemble_results.values())
plt.ylabel('Test RMSE')
plt.title('Ensemble vs Single Model')
plt.xticks(rotation=45)

# 10. Feature category analysis
plt.subplot(3, 4, 10)
feature_categories = {
    'Temporal': [f for f in selected_features if any(x in f.lower() for x in ['hour', 'day', 'month', 'time', 'sin', 'cos'])],
    'Spatial': [f for f in selected_features if any(x in f.lower() for x in ['lat', 'lon', 'urban', 'distance', 'coastal'])],
    'Emission': [f for f in selected_features if any(x in f.lower() for x in ['traffic', 'industrial', 'heating', 'emission'])],
    'Atmospheric': [f for f in selected_features if any(x in f.lower() for x in ['solar', 'inversion', 'mixing', 'stability'])],
    'Other': []
}

# Assign remaining features to 'Other'
assigned_features = sum(feature_categories.values(), [])
feature_categories['Other'] = [f for f in selected_features if f not in assigned_features]

category_counts = {k: len(v) for k, v in feature_categories.items() if len(v) > 0}
plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
plt.title('Selected Features by Category')

plt.tight_layout()
plt.savefig('d:/competition/air pollution/phase 5/comprehensive_model_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# =====================================
# 10. SUBMISSION GENERATION
# =====================================

print("\n" + "="*50)
print("GENERATING SUBMISSION")
print("="*50)

# Load test data
test_df = pd.read_csv('d:/competition/air pollution/phase 1/test.csv')
print(f"Test data shape: {test_df.shape}")

# Apply same feature engineering
test_enhanced = create_advanced_features(test_df, is_training=False)
X_submission = test_enhanced[feature_columns].fillna(0)

# Apply feature selection
X_submission_selected = selector.transform(X_submission)

# Generate predictions using weighted ensemble
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

if best_model_name in ['SVR', 'KNN', 'Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
    scaler = StandardScaler()
    scaler.fit(X_train_selected)
    X_submission_scaled = scaler.transform(X_submission_selected)
    submission_predictions = best_model.predict(X_submission_scaled)
else:
    submission_predictions = best_model.predict(X_submission_selected)

# Ensure no negative predictions
submission_predictions = np.clip(submission_predictions, 0, np.percentile(y_train, 99.5))

# Create submission file
submission_df = pd.DataFrame({
    'id': range(len(submission_predictions)),
    'pollution_value': submission_predictions
})

submission_df.to_csv('d:/competition/air pollution/phase 5/comprehensive_multi_model_submission.csv', index=False)
print("Submission file saved!")

# =====================================
# 11. FINAL SUMMARY
# =====================================

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("="*80)

print(f"\n📊 FEATURE ENGINEERING:")
print(f"• Total features created: {len(feature_columns)}")
print(f"• Features selected: {len(selected_features)}")
print(f"• Feature categories: Temporal, Spatial, Emission, Atmospheric")

print(f"\n🤖 MODELS TESTED:")
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    print(f"{i:2d}. {row['Model']:20s} - RMSE: {row['Test_RMSE']:.4f}")

print(f"\n🏆 BEST PERFORMANCE:")
print(f"• Best single model: {results_df.iloc[0]['Model']}")
print(f"• Best RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}")
print(f"• Best R²: {results_df.iloc[0]['Test_R2']:.4f}")

print(f"\n📁 FILES GENERATED:")
print(f"• comprehensive_multi_model_submission.csv")
print(f"• comprehensive_model_analysis.png")

print("="*80)
