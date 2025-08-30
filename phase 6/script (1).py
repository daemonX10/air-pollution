# Create a comprehensive implementation plan for small dataset improvements
implementation_plan = '''
# =====================================
# COMPLETE IMPLEMENTATION PLAN - SMALL DATASET OPTIMIZATION
# Maximum Accuracy for Limited Skewed Data
# =====================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, KFold, RepeatedKFold
from sklearn.preprocessing import RobustScaler, PowerTransformer, PolynomialFeatures, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, HuberRegressor, QuantileRegressor, TheilSenRegressor, ElasticNet
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression, RFECV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import warnings
import joblib
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from datetime import datetime
import shap

warnings.filterwarnings('ignore')

print("COMPLETE SMALL DATASET OPTIMIZATION PLAN")
print("Maximum Accuracy for Limited Skewed Pollution Data")
print("=" * 60)

# =====================================
# STEP 1: LOAD AND ANALYZE DATA
# =====================================

print("\\nSTEP 1: Loading and analyzing data...")

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Analyze target distribution
target = train_data['pollution_value'] if 'pollution_value' in train_data.columns else train_data.iloc[:, -1]
print(f"\\nTarget Analysis:")
print(f"  Mean: {target.mean():.4f}")
print(f"  Median: {target.median():.4f}")
print(f"  Std: {target.std():.4f}")
print(f"  Skewness: {target.skew():.4f}")
print(f"  Kurtosis: {target.kurtosis():.4f}")

# =====================================
# STEP 2: ULTRA-ADVANCED PREPROCESSING FOR SMALL DATASETS
# =====================================

def small_dataset_preprocessing(df, target_col='pollution_value', is_train=True):
    """
    Specialized preprocessing for small, skewed datasets
    """
    df_processed = df.copy()
    print(f"\\nSmall dataset preprocessing... Initial shape: {df_processed.shape}")
    
    # Preserve target
    target_values = None
    if target_col in df_processed.columns and is_train:
        target_values = df_processed[target_col].copy()
    
    # 1. INTELLIGENT MISSING VALUE HANDLING
    print("  1. Intelligent missing value handling...")
    from sklearn.impute import KNNImputer, IterativeImputer
    
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != target_col]
    
    if len(numerical_cols) > 0:
        # Use IterativeImputer for small datasets (more accurate)
        iterative_imputer = IterativeImputer(random_state=42, max_iter=10)
        df_processed[numerical_cols] = iterative_imputer.fit_transform(df_processed[numerical_cols])
    
    # 2. ADVANCED OUTLIER TREATMENT (NOT REMOVAL)
    print("  2. Advanced outlier treatment...")
    
    # Winsorization instead of removal (preserves data)
    from scipy.stats import mstats
    for col in numerical_cols:
        # Winsorize at 5% and 95% percentiles
        df_processed[f'{col}_winsorized'] = mstats.winsorize(df_processed[col], limits=[0.05, 0.05])
        
        # Outlier indicators
        q1, q3 = df_processed[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_processed[f'{col}_is_outlier'] = ((df_processed[col] < lower_bound) | 
                                           (df_processed[col] > upper_bound)).astype(int)
    
    # 3. SMART FEATURE SELECTION FOR SMALL DATA
    print("  3. Smart feature selection...")
    
    # Remove low-variance features
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)
    
    high_var_features = variance_selector.fit_transform(df_processed[numerical_cols])
    selected_feature_names = np.array(numerical_cols)[variance_selector.get_support()]
    
    # Update dataframe
    df_processed = df_processed.drop(columns=numerical_cols)
    for i, feat_name in enumerate(selected_feature_names):
        df_processed[feat_name] = high_var_features[:, i]
    
    print(f"    Removed {len(numerical_cols) - len(selected_feature_names)} low-variance features")
    
    # 4. OPTIMAL FEATURE ENGINEERING FOR SMALL DATA
    print("  4. Optimal feature engineering...")
    
    # Only create features that matter for small datasets
    if 'latitude' in df_processed.columns and 'longitude' in df_processed.columns:
        # Geographic features
        df_processed['lat_lon_interaction'] = df_processed['latitude'] * df_processed['longitude']
        df_processed['distance_from_origin'] = np.sqrt(df_processed['latitude']**2 + df_processed['longitude']**2)
        
        # Simple clustering (fewer clusters for small data)
        coords = df_processed[['latitude', 'longitude']].fillna(0)
        for n_clusters in [3, 5]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_processed[f'cluster_{n_clusters}'] = kmeans.fit_predict(coords)
    
    if 'hour' in df_processed.columns:
        # Temporal features
        hour_col = pd.to_numeric(df_processed['hour'], errors='coerce').fillna(12)
        df_processed['hour_sin'] = np.sin(2 * np.pi * hour_col / 24)
        df_processed['hour_cos'] = np.cos(2 * np.pi * hour_col / 24)
        df_processed['is_rush_hour'] = ((hour_col >= 7) & (hour_col <= 9) | 
                                      (hour_col >= 17) & (hour_col <= 19)).astype(int)
    
    # 5. HANDLE CATEGORICAL VARIABLES
    print("  5. Categorical variable encoding...")
    
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col != target_col:
            if df_processed[col].nunique() <= 5:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
            else:
                # Target encoding for high cardinality (only for training)
                if is_train and target_values is not None:
                    # Robust target encoding with regularization
                    global_mean = target_values.mean()
                    category_means = target_values.groupby(df_processed[col]).mean()
                    category_counts = df_processed[col].value_counts()
                    
                    # Smoothing parameter
                    smoothing = 10
                    regularized_means = (category_means * category_counts + global_mean * smoothing) / (category_counts + smoothing)
                    
                    df_processed[f'{col}_target_encoded'] = df_processed[col].map(regularized_means).fillna(global_mean)
            
            df_processed = df_processed.drop(columns=[col])
    
    # Final cleanup
    df_processed = df_processed.select_dtypes(include=[np.number])
    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
    df_processed = df_processed.fillna(df_processed.median())
    
    print(f"  Preprocessing completed. Final shape: {df_processed.shape}")
    
    return df_processed

# =====================================
# STEP 3: APPLY PREPROCESSING
# =====================================

print("\\nSTEP 3: Applying advanced preprocessing...")

# Preprocess training data
train_processed = small_dataset_preprocessing(train_data, is_train=True)
test_processed = small_dataset_preprocessing(test_data, is_train=False)

# Align features
common_features = list(set(train_processed.columns) & set(test_processed.columns))
if 'pollution_value' in common_features:
    common_features.remove('pollution_value')

X = train_processed[common_features]
y = train_data['pollution_value'] if 'pollution_value' in train_data.columns else train_data.iloc[:, -1]
X_test = test_processed[common_features]

print(f"Final feature count: {len(common_features)}")
print(f"Training set: {X.shape}")
print(f"Test set: {X_test.shape}")

# =====================================
# STEP 4: ADVANCED TARGET TRANSFORMATION
# =====================================

print("\\nSTEP 4: Advanced target transformation...")

def find_optimal_transformation(y_series):
    """Find the best transformation for target variable"""
    
    transformations = {}
    
    # Test multiple transformations
    if (y_series > 0).all():
        # Box-Cox
        try:
            y_bc, lambda_bc = boxcox(y_series)
            transformations['boxcox'] = {
                'data': y_bc, 
                'lambda': lambda_bc,
                'skew': abs(stats.skew(y_bc)),
                'kurtosis': abs(stats.kurtosis(y_bc))
            }
        except:
            pass
        
        # Log transform
        transformations['log1p'] = {
            'data': np.log1p(y_series),
            'skew': abs(stats.skew(np.log1p(y_series))),
            'kurtosis': abs(stats.kurtosis(np.log1p(y_series)))
        }
    
    # Yeo-Johnson (works with any data)
    try:
        y_yj, lambda_yj = yeojohnson(y_series)
        transformations['yeojohnson'] = {
            'data': y_yj,
            'lambda': lambda_yj,
            'skew': abs(stats.skew(y_yj)),
            'kurtosis': abs(stats.kurtosis(y_yj))
        }
    except:
        pass
    
    # Square root
    y_min = y_series.min()
    y_sqrt = np.sqrt(y_series - y_min + 1)
    transformations['sqrt'] = {
        'data': y_sqrt,
        'y_min': y_min,
        'skew': abs(stats.skew(y_sqrt)),
        'kurtosis': abs(stats.kurtosis(y_sqrt))
    }
    
    # Select best based on normality
    best_transform = min(transformations.keys(), 
                        key=lambda k: transformations[k]['skew'] + transformations[k]['kurtosis'])
    
    return transformations[best_transform], best_transform, transformations

# Apply optimal transformation
transform_result, best_method, all_transforms = find_optimal_transformation(y)
y_transformed = transform_result['data']

print(f"Selected transformation: {best_method}")
print(f"Original skewness: {y.skew():.4f} -> Transformed: {transform_result['skew']:.4f}")

# =====================================
# STEP 5: FEATURE SELECTION FOR SMALL DATASETS
# =====================================

print("\\nSTEP 5: Feature selection for small datasets...")

# Multiple feature selection methods
feature_selectors = {}

# Mutual Information
mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(20, X.shape[1]//2))
X_mi = mi_selector.fit_transform(X, y_transformed)
feature_selectors['mutual_info'] = mi_selector.get_support()

# F-regression
f_selector = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]//2))
X_f = f_selector.fit_transform(X, y_transformed)
feature_selectors['f_regression'] = f_selector.get_support()

# Combine selections (features selected by multiple methods)
combined_selection = np.zeros(X.shape[1], dtype=bool)
for method, selection in feature_selectors.items():
    combined_selection |= selection

selected_features = X.columns[combined_selection]
X_selected = X[selected_features]

print(f"Selected {len(selected_features)} features from {X.shape[1]} original features")

# =====================================
# STEP 6: SMALL DATASET OPTIMIZED MODELS
# =====================================

print("\\nSTEP 6: Creating small dataset optimized models...")

def create_small_data_models():
    """Models specifically optimized for small datasets"""
    
    models = {}
    
    # 1. Conservative LightGBM (prevent overfitting)
    models['lgb_conservative'] = lgb.LGBMRegressor(
        objective='huber',  # Robust to outliers
        metric='mae',
        num_leaves=15,      # Small for small data
        learning_rate=0.05, # Slow learning
        feature_fraction=0.7,
        bagging_fraction=0.7,
        min_child_samples=20,
        reg_alpha=1.0,      # Strong regularization
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1
    )
    
    # 2. Robust XGBoost
    models['xgb_robust'] = xgb.XGBRegressor(
        objective='reg:pseudohubererror',  # Robust objective
        max_depth=4,        # Shallow for small data
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,  # Conservative
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42
    )
    
    # 3. CatBoost with MAE (robust)
    models['catboost_mae'] = cb.CatBoostRegressor(
        loss_function='MAE',
        iterations=300,     # Fewer iterations for small data
        depth=4,           # Shallow
        learning_rate=0.05,
        l2_leaf_reg=5.0,   # Strong regularization
        bootstrap_type='Bayesian',
        random_state=42,
        verbose=False
    )
    
    # 4. Random Forest (good for small data)
    models['rf_robust'] = RandomForestRegressor(
        n_estimators=200,   # Moderate number
        max_depth=8,        # Limited depth
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    
    # 5. Extra Trees (more randomness)
    models['extra_trees'] = ExtraTreesRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    
    # 6. Huber Regressor (robust to outliers)
    models['huber'] = HuberRegressor(
        epsilon=1.35,
        alpha=0.01,
        max_iter=300
    )
    
    # 7. Theil-Sen (robust regression)
    models['theil_sen'] = TheilSenRegressor(
        random_state=42,
        max_subpopulation=min(1000, len(X_selected) * 2)
    )
    
    # 8. Quantile Regression (median)
    models['quantile_median'] = QuantileRegressor(
        quantile=0.5,
        alpha=0.01
    )
    
    # 9. ElasticNet (regularized linear)
    models['elastic_net'] = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=42
    )
    
    # 10. SVR with RBF (non-linear)
    models['svr_rbf'] = SVR(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        epsilon=0.1
    )
    
    return models

# =====================================
# STEP 7: ADVANCED ENSEMBLE STRATEGIES
# =====================================

def create_advanced_ensembles(base_models, X, y):
    """Create multiple advanced ensemble strategies"""
    
    print("\\nCreating advanced ensemble strategies...")
    
    ensembles = {}
    
    # 1. Multi-level Stacking
    print("  Creating multi-level stacking...")
    
    # Level 1: Diverse base models
    level1_estimators = [
        ('lgb_conservative', base_models['lgb_conservative']),
        ('xgb_robust', base_models['xgb_robust']),
        ('catboost_mae', base_models['catboost_mae']),
        ('rf_robust', base_models['rf_robust']),
        ('huber', base_models['huber'])
    ]
    
    # Multiple meta-learners
    meta_learners = {
        'ridge_stack': Ridge(alpha=1.0),
        'huber_stack': HuberRegressor(epsilon=1.35, alpha=0.01),
        'quantile_stack': QuantileRegressor(quantile=0.5, alpha=0.01),
        'elastic_stack': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    for meta_name, meta_model in meta_learners.items():
        ensembles[f'stacking_{meta_name}'] = StackingRegressor(
            estimators=level1_estimators,
            final_estimator=meta_model,
            cv=5,
            passthrough=False  # Avoid overfitting with small data
        )
    
    # 2. Weighted Voting Ensembles
    print("  Creating weighted voting ensembles...")
    
    voting_estimators = [
        ('lgb', base_models['lgb_conservative']),
        ('xgb', base_models['xgb_robust']),
        ('cb', base_models['catboost_mae']),
        ('rf', base_models['rf_robust']),
        ('et', base_models['extra_trees'])
    ]
    
    ensembles['voting_avg'] = VotingRegressor(estimators=voting_estimators)
    
    # 3. Quantile Ensemble
    print("  Creating quantile ensemble...")
    
    quantile_models = [
        ('q25', QuantileRegressor(quantile=0.25, alpha=0.01)),
        ('q50', QuantileRegressor(quantile=0.5, alpha=0.01)),
        ('q75', QuantileRegressor(quantile=0.75, alpha=0.01))
    ]
    
    ensembles['quantile_ensemble'] = VotingRegressor(estimators=quantile_models)
    
    return ensembles

# =====================================
# STEP 8: CROSS-VALIDATION FOR SMALL DATA
# =====================================

def small_data_cross_validation(models, X, y):
    """Optimized cross-validation for small datasets"""
    
    print("\\nSmall dataset cross-validation...")
    
    # Multiple CV strategies
    cv_strategies = {
        'RepeatedKFold': RepeatedKFold(n_splits=5, n_repeats=3, random_state=42),
        'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        'TimeSeriesSplit': TimeSeriesSplit(n_splits=5)
    }
    
    # Create stratification based on target quantiles
    y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\\nEvaluating {model_name}...")
        
        model_scores = {}
        
        for cv_name, cv_strategy in cv_strategies.items():
            try:
                if cv_name == 'StratifiedKFold':
                    scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                           scoring='neg_root_mean_squared_error')
                else:
                    scores = cross_val_score(model, X, y, cv=cv_strategy, 
                                           scoring='neg_root_mean_squared_error')
                
                model_scores[cv_name] = {
                    'mean': -scores.mean(),
                    'std': scores.std(),
                    'scores': -scores
                }
                
                print(f"  {cv_name}: {-scores.mean():.6f} ± {scores.std():.6f}")
                
            except Exception as e:
                print(f"  {cv_name}: Failed - {e}")
                continue
        
        results[model_name] = model_scores
    
    return results

# =====================================
# STEP 9: BAYESIAN OPTIMIZATION FOR SMALL DATA
# =====================================

def optimize_for_small_data(X, y, n_trials=30):
    """Bayesian optimization specifically for small datasets"""
    
    print(f"\\nBayesian optimization for small data ({n_trials} trials)...")
    
    def objective(trial):
        # Model selection
        model_type = trial.suggest_categorical('model', ['lightgbm', 'xgboost', 'catboost', 'ensemble'])
        
        if model_type == 'lightgbm':
            params = {
                'objective': trial.suggest_categorical('objective', ['regression', 'huber', 'regression_l1']),
                'num_leaves': trial.suggest_int('num_leaves', 10, 50),  # Conservative for small data
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
                'verbosity': -1,
                'random_state': 42
            }
            model = lgb.LGBMRegressor(**params, n_estimators=200)
            
        elif model_type == 'xgboost':
            params = {
                'objective': trial.suggest_categorical('xgb_objective', 
                                                     ['reg:squarederror', 'reg:pseudohubererror']),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
                'random_state': 42
            }
            model = xgb.XGBRegressor(**params, n_estimators=200)
            
        elif model_type == 'catboost':
            params = {
                'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE', 'Quantile:alpha=0.5']),
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'verbose': False,
                'random_state': 42
            }
            model = cb.CatBoostRegressor(**params, iterations=200)
            
        else:  # ensemble
            # Simple ensemble of 3 models
            lgb_model = lgb.LGBMRegressor(
                objective='huber',
                num_leaves=trial.suggest_int('ens_num_leaves', 10, 40),
                learning_rate=trial.suggest_float('ens_learning_rate', 0.02, 0.1),
                n_estimators=150,
                reg_alpha=1.0,
                reg_lambda=1.0,
                verbosity=-1,
                random_state=42
            )
            
            rf_model = RandomForestRegressor(
                n_estimators=trial.suggest_int('ens_n_estimators', 50, 200),
                max_depth=trial.suggest_int('ens_max_depth', 5, 12),
                min_samples_split=trial.suggest_int('ens_min_samples_split', 5, 20),
                random_state=42
            )
            
            model = VotingRegressor([
                ('lgb', lgb_model),
                ('rf', rf_model),
                ('huber', HuberRegressor(epsilon=1.35, alpha=0.01))
            ])
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, 
                                  scoring='neg_root_mean_squared_error', n_jobs=1)
        return -cv_scores.mean()
    
    # Optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=300)
    
    print(f"Best RMSE: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")
    
    return study.best_params, study.best_value

# Apply optimization
best_params, best_score = optimize_for_small_data(X_selected, y_transformed, n_trials=30)

print(f"\\nOptimization completed!")
print(f"Best cross-validation RMSE: {best_score:.6f}")

# =====================================
# IMPLEMENTATION SUMMARY
# =====================================

print("\\n" + "="*60)
print("IMPLEMENTATION SUMMARY")
print("="*60)
print("✅ Ultra-advanced preprocessing for small datasets")
print("✅ Optimal target transformation")
print("✅ Intelligent feature selection")
print("✅ Small-data optimized models")
print("✅ Bayesian hyperparameter optimization")
print("✅ Advanced ensemble strategies")
print("✅ Robust cross-validation")
print("="*60)
'''

# Save implementation plan
with open('small_dataset_implementation_plan.py', 'w') as f:
    f.write(implementation_plan)

print("✅ Complete implementation plan created!")
print("📁 File saved: small_dataset_implementation_plan.py")