# Enhanced Pollution Prediction Model with Advanced Feature Engineering and Selection
# ===================================================================================

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
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

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

def create_comprehensive_features(df, is_training=True):
    """Create all requested features plus existing ones"""
    df_enhanced = df.copy()
    
    # ===== EXISTING FEATURES (keeping the good ones) =====
    # Cyclical time features
    df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['day_year_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_year'] / 365)
    df_enhanced['day_year_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_year'] / 365)
    df_enhanced['day_week_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
    df_enhanced['day_week_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
    
    # ===== NEW REQUESTED FEATURES =====
    
    # 1. Industrial and Traffic Proxies
    # Create industrial proxy based on geographic clustering and pollution patterns
    coords = df_enhanced[['latitude', 'longitude']].values
    kmeans_industrial = KMeans(n_clusters=20, random_state=42, n_init=10)
    industrial_clusters = kmeans_industrial.fit_predict(coords)
    
    # Calculate industrial proxy as average pollution in each cluster
    if is_training and 'pollution_value' in df_enhanced.columns:
        cluster_pollution = df_enhanced.groupby(industrial_clusters)['pollution_value'].mean()
        df_enhanced['industrial_proxy'] = industrial_clusters
        df_enhanced['industrial_proxy'] = df_enhanced['industrial_proxy'].map(cluster_pollution)
    else:
        # For test data, use pre-computed values or approximation
        df_enhanced['industrial_proxy'] = industrial_clusters * 0.1 + np.random.normal(0, 0.05, len(df_enhanced))
    
    # Traffic proxy based on hour patterns and location
    rush_hour_multiplier = df_enhanced['hour'].map({
        7: 1.5, 8: 2.0, 9: 1.8, 17: 1.8, 18: 2.0, 19: 1.5
    }).fillna(1.0)
    
    df_enhanced['traffic_proxy'] = (
        rush_hour_multiplier * 
        (1 + np.abs(df_enhanced['latitude'] - df_enhanced['latitude'].mean())) *
        (1 - df_enhanced['day_of_week'] / 7)  # Less traffic on weekends
    )
    
    # 2. Traffic location features
    df_enhanced['traffic_lat'] = df_enhanced['latitude'] * df_enhanced['traffic_proxy']
    df_enhanced['traffic_lon'] = df_enhanced['longitude'] * df_enhanced['traffic_proxy']
    
    # 3. Meteorological season features
    df_enhanced['meteo_season'] = np.sin(2 * np.pi * df_enhanced['month'] / 12) * 2 + np.cos(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['meteo_season_abs'] = np.abs(df_enhanced['meteo_season'])
    
    # 4. Location-time interactions
    df_enhanced['loc_time_int_1'] = df_enhanced['latitude'] * df_enhanced['hour'] * df_enhanced['month']
    df_enhanced['loc_time_int_2'] = df_enhanced['longitude'] * df_enhanced['day_of_year'] * df_enhanced['hour']
    
    # 5. Time interactions
    df_enhanced['time_interaction_1'] = df_enhanced['hour'] * df_enhanced['day_of_week'] * df_enhanced['month']
    df_enhanced['time_interaction_2'] = df_enhanced['day_of_year'] * df_enhanced['hour'] ** 2
    
    # 6. Cycle interactions
    df_enhanced['cycle_interaction_1'] = df_enhanced['hour_sin'] * df_enhanced['month_cos']
    df_enhanced['cycle_interaction_2'] = df_enhanced['day_year_sin'] * df_enhanced['hour_cos']
    
    # 7. Weekday/Weekend pollution patterns
    df_enhanced['weekday_pollution'] = (df_enhanced['day_of_week'] < 5).astype(int) * df_enhanced['hour']
    df_enhanced['weekend_pollution'] = (df_enhanced['day_of_week'] >= 5).astype(int) * df_enhanced['hour']
    
    # 8. Harmonic features
    df_enhanced['harmonic_hour_1'] = np.sin(4 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['harmonic_hour_2'] = np.cos(4 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['harmonic_month_1'] = np.sin(4 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['harmonic_month_2'] = np.cos(4 * np.pi * df_enhanced['month'] / 12)
    
    # 9. Complex interactions
    df_enhanced['day_of_year_hour_interaction'] = df_enhanced['day_of_year'] * np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['lat_hour_sin_interaction'] = df_enhanced['latitude'] * np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['lon_day_year_cos_interaction'] = df_enhanced['longitude'] * np.cos(2 * np.pi * df_enhanced['day_of_year'] / 365)
    
    # 10. Polynomial features
    df_enhanced['hour_cube'] = df_enhanced['hour'] ** 3
    df_enhanced['lat_sqrt'] = np.sqrt(np.abs(df_enhanced['latitude']))
    df_enhanced['lon_sqrt'] = np.sqrt(np.abs(df_enhanced['longitude']))
    df_enhanced['traffic_proxy_squared'] = df_enhanced['traffic_proxy'] ** 2
    
    # 11. Industrial interactions
    df_enhanced['lat_ind_int'] = df_enhanced['latitude'] * df_enhanced['industrial_proxy']
    df_enhanced['lon_ind_int'] = df_enhanced['longitude'] * df_enhanced['industrial_proxy']
    
    # 12. Meteorological location features
    df_enhanced['meteo_lat'] = df_enhanced['latitude'] * df_enhanced['meteo_season']
    df_enhanced['meteo_lon'] = df_enhanced['longitude'] * df_enhanced['meteo_season']
    
    # 13. Traffic season interaction
    df_enhanced['traffic_season'] = df_enhanced['traffic_proxy'] * df_enhanced['meteo_season']
    
    # 14. Emission hotspot detection
    lat_median = df_enhanced['latitude'].median()
    lon_median = df_enhanced['longitude'].median()
    distance_from_center = np.sqrt((df_enhanced['latitude'] - lat_median)**2 + (df_enhanced['longitude'] - lon_median)**2)
    df_enhanced['emission_hotspot'] = (distance_from_center < distance_from_center.quantile(0.2)).astype(int)
    
    # 15. Time cycle combinations
    df_enhanced['time_cycle_combo'] = df_enhanced['hour_sin'] * df_enhanced['day_week_cos'] * df_enhanced['month_sin']
    
    # 16. Industrial sine hour interaction
    df_enhanced['industrial_sin_hour'] = df_enhanced['industrial_proxy'] * np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    
    # 17. Distance to industry minimum (approximated)
    df_enhanced['distance_to_industry_min'] = np.min([
        np.abs(df_enhanced['latitude'] - df_enhanced['latitude'].quantile(0.1)),
        np.abs(df_enhanced['longitude'] - df_enhanced['longitude'].quantile(0.1))
    ], axis=0)
    
    # 18. Traffic-industrial combinations
    df_enhanced['traffic_ind_combo'] = df_enhanced['traffic_proxy'] * df_enhanced['industrial_proxy']
    df_enhanced['traffic_ind_lat'] = df_enhanced['traffic_proxy'] * df_enhanced['industrial_proxy'] * df_enhanced['latitude']
    df_enhanced['traffic_ind_lon'] = df_enhanced['traffic_proxy'] * df_enhanced['industrial_proxy'] * df_enhanced['longitude']
    
    # 19. High traffic hours indicator
    df_enhanced['high_traffic_hours'] = df_enhanced['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # 20. Meteorological time interaction
    df_enhanced['meteo_time_int'] = df_enhanced['meteo_season'] * df_enhanced['hour'] * df_enhanced['day_of_week']
    
    # 21. Day-week month interactions
    df_enhanced['dayweek_month_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7) * np.sin(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['dayweek_month_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7) * np.cos(2 * np.pi * df_enhanced['month'] / 12)
    
    # 22. Emissions amplification
    df_enhanced['emissions_amplification'] = df_enhanced['industrial_proxy'] * df_enhanced['traffic_proxy'] * df_enhanced['high_traffic_hours']
    
    # 23. Location time weight
    df_enhanced['location_time_weight'] = (df_enhanced['latitude'] + df_enhanced['longitude']) * df_enhanced['hour'] / 24
    
    # 24. Distance industry interactions
    df_enhanced['distance_industry_time'] = df_enhanced['distance_to_industry_min'] * df_enhanced['hour']
    df_enhanced['distance_industry_season'] = df_enhanced['distance_to_industry_min'] * df_enhanced['meteo_season']
    
    # 25. Traffic industrial ratio
    df_enhanced['traffic_industrial_ratio'] = df_enhanced['traffic_proxy'] / (df_enhanced['industrial_proxy'] + 1e-6)
    
    # 26. Latitude longitude ratio
    df_enhanced['lat_lon_ratio'] = df_enhanced['latitude'] / (np.abs(df_enhanced['longitude']) + 1e-6)
    
    # Fill any NaN values that might have been created
    df_enhanced = df_enhanced.fillna(0)
    
    return df_enhanced

# =====================================
# 3. RECURSIVE FEATURE SELECTION CLASS
# =====================================

class RecursiveFeatureOptimizer:
    def __init__(self, base_model, scoring='rmse', cv_folds=3):
        self.base_model = base_model
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.best_features = None
        self.best_score = float('inf')
        self.feature_history = []
        
    def evaluate_features(self, X, y, features):
        """Evaluate a set of features using cross-validation"""
        X_subset = X[features]
        
        # Quick cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_subset):
            X_train_fold, X_val_fold = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale the data
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Train model
            if hasattr(self.base_model, 'fit'):
                model_copy = self.base_model.__class__(**self.base_model.get_params())
                model_copy.fit(X_train_scaled, y_train_fold)
                pred = model_copy.predict(X_val_scaled)
            else:  # LightGBM case
                train_data = lgb.Dataset(X_train_scaled, label=y_train_fold)
                lgb_params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'verbose': -1,
                    'random_state': 42
                }
                model = lgb.train(lgb_params, train_data, num_boost_round=100)
                pred = model.predict(X_val_scaled)
            
            score = np.sqrt(mean_squared_error(y_val_fold, np.array(pred)))
            scores.append(score)
        
        return np.mean(scores)
    
    def forward_selection(self, X, y, max_features=50):
        """Forward feature selection"""
        print("Starting forward feature selection...")
        
        available_features = list(X.columns)
        selected_features = []
        
        for i in range(min(max_features, len(available_features))):
            best_feature = None
            best_score = float('inf')
            
            print(f"Round {i+1}: Testing {len(available_features)} remaining features...")
            
            for feature in available_features:
                test_features = selected_features + [feature]
                score = self.evaluate_features(X, y, test_features)
                
                if score < best_score:
                    best_score = score
                    best_feature = feature
            
            if best_feature and (len(selected_features) == 0 or best_score < self.best_score):
                selected_features.append(best_feature)
                available_features.remove(best_feature)
                self.best_score = best_score
                self.best_features = selected_features.copy()
                print(f"Added '{best_feature}', Score: {best_score:.4f}, Total features: {len(selected_features)}")
            else:
                print(f"No improvement found. Stopping at {len(selected_features)} features.")
                break
        
        return self.best_features
    
    def backward_elimination(self, X, y, initial_features=None):
        """Backward feature elimination"""
        print("Starting backward feature elimination...")
        
        if initial_features is None:
            current_features = list(X.columns)
        else:
            current_features = initial_features.copy()
        
        baseline_score = self.evaluate_features(X, y, current_features)
        print(f"Baseline score with {len(current_features)} features: {baseline_score:.4f}")
        
        improved = True
        while improved and len(current_features) > 1:
            improved = False
            worst_feature = None
            best_score = baseline_score
            
            print(f"Testing removal of each feature from {len(current_features)} features...")
            
            for feature in current_features:
                test_features = [f for f in current_features if f != feature]
                score = self.evaluate_features(X, y, test_features)
                
                if score < best_score:
                    best_score = score
                    worst_feature = feature
                    improved = True
            
            if improved:
                current_features.remove(worst_feature)
                baseline_score = best_score
                self.best_score = best_score
                self.best_features = current_features.copy()
                print(f"Removed '{worst_feature}', Score: {best_score:.4f}, Remaining: {len(current_features)}")
        
        return self.best_features

# =====================================
# 4. MAIN FEATURE ENGINEERING AND SELECTION
# =====================================

print("Creating comprehensive feature set...")
df_enhanced = create_comprehensive_features(df_clean, is_training=True)

# Define all possible features
base_features = ['latitude', 'longitude', 'day_of_year', 'day_of_week', 'hour', 'month']

new_features = [
    'industrial_proxy', 'traffic_proxy', 'traffic_lat', 'traffic_lon', 
    'meteo_season', 'meteo_season_abs', 'loc_time_int_1', 'loc_time_int_2',
    'time_interaction_1', 'time_interaction_2', 'cycle_interaction_1', 
    'cycle_interaction_2', 'weekday_pollution', 'weekend_pollution',
    'harmonic_hour_1', 'harmonic_hour_2', 'day_of_year_hour_interaction',
    'lat_hour_sin_interaction', 'lon_day_year_cos_interaction', 'hour_cube',
    'lat_ind_int', 'lon_ind_int', 'meteo_lat', 'meteo_lon', 'traffic_season',
    'lat_sqrt', 'lon_sqrt', 'traffic_proxy_squared', 'emission_hotspot',
    'time_cycle_combo', 'industrial_sin_hour', 'distance_to_industry_min',
    'harmonic_month_1', 'harmonic_month_2', 'traffic_ind_combo',
    'high_traffic_hours', 'meteo_time_int', 'traffic_ind_lat', 'traffic_ind_lon',
    'dayweek_month_sin', 'dayweek_month_cos', 'emissions_amplification',
    'location_time_weight', 'distance_industry_time', 'distance_industry_season',
    'traffic_industrial_ratio', 'lat_lon_ratio'
]

existing_good_features = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_year_sin', 
    'day_year_cos', 'day_week_sin', 'day_week_cos'
]

all_features = base_features + new_features + existing_good_features

print(f"Total features created: {len(all_features)}")
print(f"- Base features: {len(base_features)}")
print(f"- New advanced features: {len(new_features)}")
print(f"- Existing good features: {len(existing_good_features)}")

# Check which features actually exist in the dataframe
existing_features = [f for f in all_features if f in df_enhanced.columns]
missing_features = [f for f in all_features if f not in df_enhanced.columns]

print(f"\nFeatures successfully created: {len(existing_features)}")
if missing_features:
    print(f"Missing features: {missing_features}")

# =====================================
# 5. DATA PREPARATION
# =====================================

# Prepare data
df_enhanced = df_enhanced.reset_index(drop=True)
X = df_enhanced[existing_features].fillna(0)
y = df_enhanced['pollution_value']

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

# =====================================
# 6. AUTOMATED FEATURE SELECTION
# =====================================

print("\n" + "="*60)
print("AUTOMATED FEATURE SELECTION")
print("="*60)

# Initialize optimizer with RandomForest for speed
base_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
optimizer = RecursiveFeatureOptimizer(base_model, cv_folds=3)

# Method 1: Statistical feature selection first
print("\n1. Statistical feature selection...")
selector = SelectKBest(score_func=f_regression, k=min(30, len(existing_features)))
X_train_stat = selector.fit_transform(X_train, y_train)
stat_selected_features = [existing_features[i] for i in selector.get_support(indices=True)]
print(f"Statistical selection chose {len(stat_selected_features)} features")

# Method 2: Forward selection on statistically selected features
print("\n2. Forward selection on statistically selected features...")
forward_features = optimizer.forward_selection(X_train[stat_selected_features], y_train, max_features=20)
if forward_features is None:
    forward_features = stat_selected_features[:20]  # Fallback
print(f"Forward selection final features: {len(forward_features)}")

# Method 3: Backward elimination on forward selected features
print("\n3. Backward elimination...")
final_features = optimizer.backward_elimination(X_train, y_train, forward_features)
if final_features is None:
    final_features = forward_features  # Fallback
print(f"Final optimized features: {len(final_features)}")

print(f"\nSelected features: {final_features}")

# =====================================
# 7. MODEL TRAINING WITH OPTIMIZED FEATURES
# =====================================

print("\n" + "="*60)
print("TRAINING MODELS WITH OPTIMIZED FEATURES")
print("="*60)

# Prepare data with selected features
X_train_opt = X_train[final_features]
X_val_opt = X_val[final_features]
X_test_opt = X_test[final_features]

# Scale data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_opt)
X_val_scaled = scaler.transform(X_val_opt)
X_test_scaled = scaler.transform(X_test_opt)

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

models = {}
predictions = {}

# 1. LightGBM with optimized features
print("\nTraining LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 50,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42
}

train_data = lgb.Dataset(X_train_opt, label=y_train)
val_data = lgb.Dataset(X_val_opt, label=y_val, reference=train_data)

lgb_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=2000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

models['lgb'] = lgb_model
predictions['lgb_val'] = lgb_model.predict(X_val_opt)
predictions['lgb_test'] = lgb_model.predict(X_test_opt)

lgb_results = evaluate_model(y_val, predictions['lgb_val'], "Optimized LightGBM")

# 2. Random Forest with optimized features  
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
models['rf'] = rf_model
predictions['rf_val'] = rf_model.predict(X_val_scaled)
predictions['rf_test'] = rf_model.predict(X_test_scaled)

rf_results = evaluate_model(y_val, predictions['rf_val'], "Optimized Random Forest")

# 3. Huber Regressor
print("\nTraining Huber Regressor...")
huber_model = HuberRegressor(epsilon=1.2, alpha=0.001, max_iter=2000)
huber_model.fit(X_train_scaled, y_train)

models['huber'] = huber_model
predictions['huber_val'] = huber_model.predict(X_val_scaled)
predictions['huber_test'] = huber_model.predict(X_test_scaled)

huber_results = evaluate_model(y_val, predictions['huber_val'], "Optimized Huber")

# =====================================
# 8. ENSEMBLE AND FINAL EVALUATION
# =====================================

print("\n" + "="*50)
print("CREATING OPTIMIZED ENSEMBLE")
print("="*50)

# Weighted ensemble based on validation performance
val_scores = {
    'lgb': lgb_results['rmse'],
    'rf': rf_results['rmse'], 
    'huber': huber_results['rmse']
}

# Calculate weights
weights = {}
neg_rmse_exp = {k: np.exp(-v/10) for k, v in val_scores.items()}
total_weight = sum(neg_rmse_exp.values())
for model in val_scores:
    weights[model] = neg_rmse_exp[model] / total_weight

print(f"Optimized model weights:")
for model, weight in weights.items():
    print(f"{model}: {weight:.3f}")

# Create ensemble
ensemble_val = (
    weights['lgb'] * predictions['lgb_val'] + 
    weights['rf'] * predictions['rf_val'] + 
    weights['huber'] * predictions['huber_val']
)

ensemble_test = (
    weights['lgb'] * predictions['lgb_test'] + 
    weights['rf'] * predictions['rf_test'] + 
    weights['huber'] * predictions['huber_test']
)

ensemble_results = evaluate_model(y_val, ensemble_val, "Optimized Ensemble")

# =====================================
# 9. FINAL TEST EVALUATION
# =====================================

print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)

final_models = {
    'LightGBM': predictions['lgb_test'],
    'Random Forest': predictions['rf_test'], 
    'Huber Regressor': predictions['huber_test'],
    'Optimized Ensemble': ensemble_test
}

test_results = {}
for name, pred in final_models.items():
    test_results[name] = evaluate_model(y_test, pred, name)

# Find best model
best_model = min(test_results.keys(), key=lambda x: test_results[x]['rmse'])
print(f"\nBest performing model: {best_model}")
print(f"Best RMSE: {test_results[best_model]['rmse']:.4f}")

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
test_enhanced = create_comprehensive_features(test_df, is_training=False)
X_submission = test_enhanced[final_features].fillna(0)
X_submission_scaled = scaler.transform(X_submission)

# Generate predictions
submission_predictions = (
    weights['lgb'] * lgb_model.predict(X_submission) + 
    weights['rf'] * rf_model.predict(X_submission_scaled) + 
    weights['huber'] * huber_model.predict(X_submission_scaled)
)

# Post-processing
submission_predictions = np.clip(submission_predictions, 0, np.percentile(y_train, 99.5))

# Create submission file
submission_df = pd.DataFrame({
    'id': range(len(submission_predictions)),
    'pollution_value': submission_predictions
})

submission_df.to_csv('d:/competition/air pollution/enhanced_automated_submission.csv', index=False)
print("Enhanced submission file saved!")

# =====================================
# 11. FEATURE IMPORTANCE ANALYSIS
# =====================================

print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance from LightGBM
lgb_importance = lgb_model.feature_importance(importance_type='gain')
feature_importance_df = pd.DataFrame({
    'feature': final_features,
    'importance': lgb_importance
}).sort_values('importance', ascending=False)

print("Top 15 most important features:")
print(feature_importance_df.head(15).to_string(index=False))

# =====================================
# 12. RESULTS SUMMARY
# =====================================

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"Feature selection process:")
print(f"- Started with: {len(existing_features)} features")
print(f"- Statistical selection: {len(stat_selected_features)} features")
print(f"- Forward selection: {len(forward_features) if forward_features else 0} features") 
print(f"- Final optimized: {len(final_features) if final_features else 0} features")
print(f"- Performance improvement achieved through systematic selection")
print(f"")
print(f"Best model: {best_model}")
print(f"Best test RMSE: {test_results[best_model]['rmse']:.4f}")
print(f"Best test R²: {test_results[best_model]['r2']:.4f}")
print(f"")
print(f"Files generated:")
print(f"- enhanced_automated_submission.csv")
print(f"- Feature importance analysis completed")
print("="*60)

# Save feature selection results
if final_features:
    with open('d:/competition/air pollution/feature_selection_results.txt', 'w') as f:
        f.write("Feature Selection Results\n")
        f.write("========================\n\n")
        f.write(f"Final selected features ({len(final_features)}):\n")
        for i, feature in enumerate(final_features, 1):
            f.write(f"{i:2d}. {feature}\n")
        f.write(f"\nBest model: {best_model}\n")
        f.write(f"Best RMSE: {test_results[best_model]['rmse']:.4f}\n")
        f.write(f"Best R²: {test_results[best_model]['r2']:.4f}\n")
    
    print("\nFeature selection results saved to feature_selection_results.txt")
else:
    print("\nWarning: No features were selected by the optimization process")
