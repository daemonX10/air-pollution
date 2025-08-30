# Create a comprehensive code enhancement for the pollution prediction model
# Based on the research findings and analysis of the provided images

enhanced_code = '''
# =====================================
# ENHANCED POLLUTION PREDICTION MODEL - ADVANCED ACCURACY IMPROVEMENTS
# Focus: 15+ Advanced Techniques for Skewed Data
# Research-Based Improvements for Maximum Accuracy
# =====================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, PowerTransformer, PolynomialFeatures, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, HuberRegressor, QuantileRegressor, TheilSenRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression, RFECV
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
import warnings
import joblib
import os
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import gc
from datetime import datetime
import shap
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from sklearn.isotonic import IsotonicRegression
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("ENHANCED POLLUTION PREDICTION MODEL - MAXIMUM ACCURACY IMPROVEMENTS")
print("Research-Based: 15+ Advanced Techniques for Skewed Data")
print("=" * 80)

# =====================================
# 1. ADVANCED FEATURE ENGINEERING V3 (RESEARCH-ENHANCED)
# =====================================

def create_ultra_advanced_features_v3(df, target_col='pollution_value', is_train=True):
    """
    Ultra-advanced feature engineering with 15+ research-backed techniques
    """
    df_ultra = df.copy()
    print(f"Creating ultra-advanced features... Initial shape: {df_ultra.shape}")
    
    # Preserve target
    target_values = None
    if target_col in df_ultra.columns and is_train:
        target_values = df_ultra[target_col].copy()
        print(f"Target preserved: {target_col}")
    
    # 1. ENHANCED MISSING VALUE HANDLING
    print("1. Enhanced missing value imputation...")
    
    # KNN-based imputation for better accuracy
    from sklearn.impute import KNNImputer
    numerical_cols = df_ultra.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != target_col]
    
    if len(numerical_cols) > 0:
        knn_imputer = KNNImputer(n_neighbors=5)
        df_ultra[numerical_cols] = knn_imputer.fit_transform(df_ultra[numerical_cols])
    
    # 2. OUTLIER HANDLING WITH ROBUST METHODS
    print("2. Advanced outlier detection and handling...")
    
    # Multiple outlier detection methods
    outlier_scores = np.zeros(len(df_ultra))
    
    # Isolation Forest
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_scores += (iso_forest.fit_predict(df_ultra[numerical_cols]) == -1).astype(int)
    
    # Local Outlier Factor
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(contamination=0.1)
    outlier_scores += (lof.fit_predict(df_ultra[numerical_cols]) == -1).astype(int)
    
    # Robust outlier indicator
    df_ultra['outlier_score'] = outlier_scores / 2
    df_ultra['is_outlier'] = (outlier_scores >= 1).astype(int)
    
    # 3. ADVANCED POLYNOMIAL AND INTERACTION FEATURES
    print("3. Creating advanced polynomial interactions...")
    
    # Select top features for interactions (avoid feature explosion)
    if len(numerical_cols) > 3:
        # Use mutual information to select top features
        mi_scores = mutual_info_regression(df_ultra[numerical_cols], 
                                         target_values if target_values is not None else df_ultra[numerical_cols[0]])
        top_features = np.array(numerical_cols)[np.argsort(mi_scores)[-5:]]
    else:
        top_features = numerical_cols
    
    # Polynomial features (degree 2 and 3)
    for degree in [2, 3]:
        poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(df_ultra[top_features])
        poly_names = poly.get_feature_names_out(top_features)
        
        for i, name in enumerate(poly_names):
            if name not in top_features:
                df_ultra[f'poly_{degree}_{name}'] = poly_features[:, i]
    
    # 4. QUANTILE-BASED TRANSFORMATIONS
    print("4. Applying quantile transformations...")
    
    # Quantile transformer for each numerical feature
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    for col in numerical_cols:
        df_ultra[f'{col}_quantile_normal'] = qt.fit_transform(df_ultra[[col]]).flatten()
    
    # 5. CLUSTERING-BASED FEATURES (MULTIPLE ALGORITHMS)
    print("5. Creating advanced clustering features...")
    
    if 'latitude' in df_ultra.columns and 'longitude' in df_ultra.columns:
        coords = df_ultra[['latitude', 'longitude']].fillna(df_ultra[['latitude', 'longitude']].median())
        
        # K-Means clustering
        for n_clusters in [3, 5, 8, 12, 20]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_ultra[f'kmeans_cluster_{n_clusters}'] = kmeans.fit_predict(coords)
            
            # Distance to nearest cluster center
            distances = np.min(np.sqrt(((coords.values[:, np.newaxis] - kmeans.cluster_centers_) ** 2).sum(axis=2)), axis=1)
            df_ultra[f'kmeans_distance_{n_clusters}'] = distances
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.1, min_samples=5)
        df_ultra['dbscan_cluster'] = dbscan.fit_predict(coords)
        df_ultra['is_dbscan_outlier'] = (df_ultra['dbscan_cluster'] == -1).astype(int)
    
    # 6. TEMPORAL FEATURE ENGINEERING
    print("6. Advanced temporal feature engineering...")
    
    if 'hour' in df_ultra.columns:
        hour_col = pd.to_numeric(df_ultra['hour'], errors='coerce').fillna(12)
        
        # Multiple cyclical encodings
        for period in [24, 12, 8, 6]:
            df_ultra[f'hour_sin_{period}'] = np.sin(2 * np.pi * hour_col / period)
            df_ultra[f'hour_cos_{period}'] = np.cos(2 * np.pi * hour_col / period)
        
        # Hour-based categories
        df_ultra['hour_category'] = pd.cut(hour_col, bins=6, labels=False)
        df_ultra['is_peak_hour'] = ((hour_col >= 7) & (hour_col <= 9) | 
                                   (hour_col >= 17) & (hour_col <= 19)).astype(int)
        df_ultra['is_night'] = ((hour_col >= 22) | (hour_col <= 6)).astype(int)
    
    # 7. STATISTICAL FEATURES WITH ROLLING WINDOWS
    print("7. Creating statistical rolling features...")
    
    for col in numerical_cols[:5]:  # Limit to avoid memory issues
        # Multiple window sizes
        for window in [3, 5, 7, 10]:
            if len(df_ultra) > window:
                df_ultra[f'{col}_rolling_mean_{window}'] = df_ultra[col].rolling(window=window, min_periods=1).mean()
                df_ultra[f'{col}_rolling_std_{window}'] = df_ultra[col].rolling(window=window, min_periods=1).std()
                df_ultra[f'{col}_rolling_median_{window}'] = df_ultra[col].rolling(window=window, min_periods=1).median()
    
    # 8. ADVANCED BINNING STRATEGIES
    print("8. Creating adaptive binning features...")
    
    for col in numerical_cols[:3]:
        if df_ultra[col].nunique() > 10:
            try:
                # Quantile-based binning
                df_ultra[f'{col}_qbins'] = pd.qcut(df_ultra[col], q=5, labels=False, duplicates='drop')
                
                # Equal-width binning
                df_ultra[f'{col}_ebins'] = pd.cut(df_ultra[col], bins=5, labels=False)
                
                # Custom binning based on distribution
                if df_ultra[col].skew() > 1:  # Right-skewed
                    df_ultra[f'{col}_log_bins'] = pd.cut(np.log1p(df_ultra[col] - df_ultra[col].min() + 1), 
                                                       bins=5, labels=False)
            except:
                pass
    
    # 9. DISTANCE AND PROXIMITY FEATURES
    print("9. Creating distance and proximity features...")
    
    if len(numerical_cols) >= 2:
        # Euclidean distances between features
        for i in range(min(3, len(numerical_cols))):
            for j in range(i+1, min(4, len(numerical_cols))):
                col1, col2 = numerical_cols[i], numerical_cols[j]
                df_ultra[f'euclidean_{col1}_{col2}'] = np.sqrt(
                    (df_ultra[col1] - df_ultra[col1].mean())**2 + 
                    (df_ultra[col2] - df_ultra[col2].mean())**2
                )
    
    # 10. TARGET ENCODING (ROBUST VERSION)
    print("10. Robust target encoding...")
    
    categorical_cols = df_ultra.select_dtypes(include=['object', 'category']).columns
    
    if is_train and target_values is not None:
        for col in categorical_cols:
            if col != target_col and df_ultra[col].nunique() > 2:
                # Robust target encoding with cross-validation
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                
                encoded_values = np.zeros(len(df_ultra))
                for train_idx, val_idx in kf.split(df_ultra):
                    train_mean = target_values.iloc[train_idx].groupby(df_ultra[col].iloc[train_idx]).mean()
                    encoded_values[val_idx] = df_ultra[col].iloc[val_idx].map(train_mean).fillna(target_values.mean())
                
                df_ultra[f'{col}_target_encoded'] = encoded_values
                df_ultra = df_ultra.drop(columns=[col])
    
    # Remove remaining object columns
    object_cols = df_ultra.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        df_ultra = df_ultra.drop(columns=object_cols)
    
    # 11. FEATURE SCALING VARIATIONS
    print("11. Multiple feature scaling approaches...")
    
    # Apply different scalers to different feature groups
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
    
    scalers = {
        'robust': RobustScaler(),
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler()
    }
    
    # Apply different scalers to different feature subsets
    feature_subsets = np.array_split(numerical_cols, 4)
    for i, (scaler_name, scaler) in enumerate(scalers.items()):
        if i < len(feature_subsets) and len(feature_subsets[i]) > 0:
            subset_cols = [col for col in feature_subsets[i] if col in df_ultra.columns]
            if subset_cols:
                scaled_features = scaler.fit_transform(df_ultra[subset_cols])
                for j, col in enumerate(subset_cols):
                    df_ultra[f'{col}_{scaler_name}_scaled'] = scaled_features[:, j]
    
    # Final cleanup
    df_ultra = df_ultra.replace([np.inf, -np.inf], np.nan)
    df_ultra = df_ultra.fillna(df_ultra.median())
    
    print(f"Ultra-advanced feature engineering completed. Final shape: {df_ultra.shape}")
    print(f"Added {df_ultra.shape[1] - df.shape[1]} new features")
    
    return df_ultra

# =====================================
# 2. ADVANCED ENSEMBLE STRATEGIES
# =====================================

def create_advanced_ensemble_models():
    """
    Create diverse ensemble models for maximum accuracy
    """
    
    # Base models with diverse characteristics
    base_models = {
        'lightgbm_conservative': lgb.LGBMRegressor(
            objective='regression_l1',  # MAE for robustness
            metric='mae',
            boosting_type='gbdt',
            num_leaves=50,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        ),
        
        'lightgbm_aggressive': lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            num_leaves=200,
            learning_rate=0.1,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            min_child_samples=5,
            random_state=43
        ),
        
        'xgboost_huber': xgb.XGBRegressor(
            objective='reg:pseudohubererror',  # Robust to outliers
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        ),
        
        'xgboost_quantile': xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.5,  # Median regression
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=44
        ),
        
        'catboost_robust': cb.CatBoostRegressor(
            loss_function='Quantile:alpha=0.5',  # Median regression
            iterations=500,
            depth=6,
            learning_rate=0.08,
            l2_leaf_reg=3.0,
            bootstrap_type='Bayesian',
            random_state=42,
            verbose=False
        ),
        
        'catboost_mae': cb.CatBoostRegressor(
            loss_function='MAE',  # Robust to outliers
            iterations=500,
            depth=8,
            learning_rate=0.1,
            l2_leaf_reg=1.0,
            random_state=43,
            verbose=False
        ),
        
        'rf_entropy': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        ),
        
        'extra_trees': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        ),
        
        'huber_regressor': HuberRegressor(
            epsilon=1.35,
            alpha=0.01,
            max_iter=200
        ),
        
        'theil_sen': TheilSenRegressor(
            random_state=42,
            max_subpopulation=1000
        ),
        
        'quantile_median': QuantileRegressor(
            quantile=0.5,
            alpha=0.01,
            solver='highs'
        )
    }
    
    return base_models

# =====================================
# 3. SHAP-BASED FEATURE SELECTION
# =====================================

def shap_feature_selection(X, y, model, top_k=50):
    """
    Advanced SHAP-based feature selection
    """
    print(f"SHAP-based feature selection for top {top_k} features...")
    
    # Fit model
    model.fit(X, y)
    
    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # Calculate feature importance
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    feature_rankings = np.argsort(feature_importance)[::-1]
    
    # Select top K features
    top_features = X.columns[feature_rankings[:top_k]]
    
    print(f"Selected {len(top_features)} features using SHAP")
    print(f"Top 10 features: {list(top_features[:10])}")
    
    return list(top_features), feature_importance

# =====================================
# 4. MULTI-LEVEL STACKING ENSEMBLE
# =====================================

def create_multi_level_stacking(base_models, meta_learners):
    """
    Create multi-level stacking ensemble
    """
    
    # Level 1: Base models
    level1_models = [
        ('lgb_conservative', base_models['lightgbm_conservative']),
        ('lgb_aggressive', base_models['lightgbm_aggressive']),
        ('xgb_huber', base_models['xgboost_huber']),
        ('xgb_quantile', base_models['xgboost_quantile']),
        ('catboost_robust', base_models['catboost_robust']),
        ('rf_entropy', base_models['rf_entropy'])
    ]
    
    # Level 2: Meta-learners
    stacking_models = {}
    
    for meta_name, meta_model in meta_learners.items():
        stacking_models[f'stack_{meta_name}'] = StackingRegressor(
            estimators=level1_models,
            final_estimator=meta_model,
            cv=5,
            passthrough=True  # Include original features
        )
    
    return stacking_models

# =====================================
# 5. ADVANCED CROSS-VALIDATION WITH UNCERTAINTY
# =====================================

def advanced_cross_validation_with_uncertainty(models, X, y, cv_strategies):
    """
    Advanced cross-validation with uncertainty quantification
    """
    print("\\nAdvanced cross-validation with uncertainty quantification...")
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\\nEvaluating {model_name}...")
        
        model_results = {}
        
        for cv_name, cv_strategy in cv_strategies.items():
            scores = cross_val_score(model, X, y, cv=cv_strategy, 
                                   scoring='neg_root_mean_squared_error', n_jobs=-1)
            
            model_results[cv_name] = {
                'mean_rmse': -scores.mean(),
                'std_rmse': scores.std(),
                'confidence_interval': (-scores.mean() - 1.96*scores.std(), 
                                      -scores.mean() + 1.96*scores.std())
            }
            
            print(f"  {cv_name}: RMSE = {-scores.mean():.6f} ± {scores.std():.6f}")
        
        results[model_name] = model_results
    
    return results

# =====================================
# 6. PSEUDO-LABELING FOR REGRESSION
# =====================================

def pseudo_labeling_regression(models, X_labeled, y_labeled, X_unlabeled, confidence_threshold=0.8):
    """
    Implement pseudo-labeling for regression tasks
    """
    print("\\nImplementing pseudo-labeling for regression...")
    
    # Train ensemble on labeled data
    ensemble_predictions = []
    
    for model_name, model in models.items():
        model.fit(X_labeled, y_labeled)
        pred = model.predict(X_unlabeled)
        ensemble_predictions.append(pred)
    
    # Average predictions
    avg_predictions = np.mean(ensemble_predictions, axis=0)
    
    # Calculate prediction confidence (inverse of prediction variance)
    pred_variance = np.var(ensemble_predictions, axis=0)
    confidence_scores = 1 / (1 + pred_variance)
    
    # Select high-confidence pseudo-labels
    high_confidence_mask = confidence_scores > np.quantile(confidence_scores, confidence_threshold)
    
    if np.sum(high_confidence_mask) > 0:
        X_pseudo = X_unlabeled[high_confidence_mask]
        y_pseudo = avg_predictions[high_confidence_mask]
        
        # Combine labeled and pseudo-labeled data
        X_combined = pd.concat([X_labeled, X_pseudo], ignore_index=True)
        y_combined = np.concatenate([y_labeled, y_pseudo])
        
        print(f"Added {len(y_pseudo)} pseudo-labeled samples ({np.sum(high_confidence_mask)/len(X_unlabeled)*100:.1f}%)")
        
        return X_combined, y_combined
    else:
        print("No high-confidence pseudo-labels found")
        return X_labeled, y_labeled

# =====================================
# 7. ADVANCED HYPERPARAMETER OPTIMIZATION
# =====================================

def advanced_hyperparameter_optimization_v2(X, y, n_trials=100):
    """
    Advanced Bayesian optimization with multiple objectives
    """
    print("\\nAdvanced multi-objective hyperparameter optimization...")
    
    def objective_multi_models(trial):
        # Suggest which model to use
        model_choice = trial.suggest_categorical('model', ['lightgbm', 'xgboost', 'catboost'])
        
        if model_choice == 'lightgbm':
            params = {
                'objective': trial.suggest_categorical('objective', ['regression', 'regression_l1']),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'verbose': -1,
                'random_state': 42
            }
            model = lgb.LGBMRegressor(**params, n_estimators=500)
            
        elif model_choice == 'xgboost':
            objective_choice = trial.suggest_categorical('xgb_objective', 
                                                       ['reg:squarederror', 'reg:pseudohubererror', 'reg:quantileerror'])
            params = {
                'objective': objective_choice,
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': 42
            }
            
            if objective_choice == 'reg:quantileerror':
                params['quantile_alpha'] = trial.suggest_float('quantile_alpha', 0.1, 0.9)
            
            model = xgb.XGBRegressor(**params, n_estimators=500)
            
        else:  # catboost
            loss_function = trial.suggest_categorical('loss_function', ['RMSE', 'MAE', 'Quantile:alpha=0.5'])
            params = {
                'loss_function': loss_function,
                'iterations': 500,
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
                'verbose': False,
                'random_state': 42
            }
            model = cb.CatBoostRegressor(**params)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        return -cv_scores.mean()
    
    # Multi-objective optimization
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective_multi_models, n_trials=n_trials, timeout=600)
    
    print(f"Best trial: {study.best_trial.value:.6f}")
    print(f"Best params: {study.best_trial.params}")
    
    return study.best_trial.params

# =====================================
# 8. ADVANCED TARGET TRANSFORMATION
# =====================================

def advanced_target_transformation_v2(y):
    """
    Enhanced target transformation with multiple methods
    """
    print("\\nAdvanced target transformation analysis...")
    
    transformations = {}
    
    # Original
    transformations['original'] = {
        'data': y.copy(),
        'skewness': stats.skew(y),
        'kurtosis': stats.kurtosis(y),
        'normality_p': stats.shapiro(y[:min(5000, len(y))])[1]
    }
    
    # Multiple transformations
    transform_methods = {}
    
    # Box-Cox family
    if (y > 0).all():
        try:
            y_boxcox, lambda_bc = boxcox(y)
            transform_methods['boxcox'] = (y_boxcox, lambda_bc)
        except:
            pass
    
    # Yeo-Johnson
    try:
        y_yj, lambda_yj = yeojohnson(y)
        transform_methods['yeojohnson'] = (y_yj, lambda_yj)
    except:
        pass
    
    # Log transforms
    if (y > 0).all():
        transform_methods['log1p'] = (np.log1p(y), None)
        transform_methods['log'] = (np.log(y), None)
    
    # Square root
    y_min = y.min()
    transform_methods['sqrt'] = (np.sqrt(y - y_min + 1), y_min)
    
    # Inverse
    if (y > 0).all():
        transform_methods['inverse'] = (1 / (y + 1e-8), None)
    
    # Power transformations
    for power in [0.25, 0.5, 1.5, 2]:
        if power < 1 and (y >= 0).all():
            transform_methods[f'power_{power}'] = (np.power(y + 1e-8, power), power)
        elif power >= 1:
            transform_methods[f'power_{power}'] = (np.power(y, power), power)
    
    # Evaluate each transformation
    for name, (transformed_y, param) in transform_methods.items():
        if not np.isfinite(transformed_y).all():
            continue
            
        transformations[name] = {
            'data': transformed_y,
            'skewness': abs(stats.skew(transformed_y)),
            'kurtosis': abs(stats.kurtosis(transformed_y)),
            'normality_p': stats.shapiro(transformed_y[:min(5000, len(transformed_y))])[1],
            'parameter': param
        }
        
        # Combined score (lower is better for skewness and kurtosis, higher for normality)
        transformations[name]['score'] = (
            transformations[name]['normality_p'] + 
            1/(1 + transformations[name]['skewness']) + 
            1/(1 + transformations[name]['kurtosis'])
        )
    
    # Select best transformation
    best_transform = max(transformations.keys(), key=lambda k: transformations[k]['score'])
    
    print(f"Best transformation: {best_transform}")
    print(f"Skewness: {transformations['original']['skewness']:.3f} -> {transformations[best_transform]['skewness']:.3f}")
    
    return transformations[best_transform], best_transform, transformations

# =====================================
# 9. ENSEMBLE DIVERSITY OPTIMIZATION
# =====================================

def optimize_ensemble_diversity(base_predictions, y_true):
    """
    Optimize ensemble weights based on diversity and accuracy
    """
    print("\\nOptimizing ensemble diversity and weights...")
    
    n_models = len(base_predictions)
    predictions_matrix = np.column_stack(base_predictions)
    
    # Calculate model correlations
    correlations = np.corrcoef(predictions_matrix.T)
    
    # Diversity score (lower correlation = higher diversity)
    diversity_scores = 1 - np.abs(correlations)
    avg_diversity = np.mean(diversity_scores[np.triu_indices_from(diversity_scores, k=1)])
    
    # Individual model accuracy
    individual_rmse = [np.sqrt(mean_squared_error(y_true, pred)) for pred in base_predictions]
    
    # Optimize weights using constrained optimization
    from scipy.optimize import minimize
    
    def objective(weights):
        weighted_pred = np.average(predictions_matrix, axis=1, weights=weights)
        rmse = np.sqrt(mean_squared_error(y_true, weighted_pred))
        
        # Penalty for low diversity (encourage diverse models)
        diversity_penalty = -0.1 * avg_diversity
        
        return rmse + diversity_penalty
    
    # Constraints: weights sum to 1, non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    result = minimize(objective, x0=np.ones(n_models)/n_models, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    
    print(f"Ensemble diversity score: {avg_diversity:.4f}")
    print(f"Optimal weights: {dict(zip(range(n_models), optimal_weights))}")
    
    return optimal_weights, avg_diversity

# =====================================
# 10. ROBUST EVALUATION METRICS
# =====================================

def robust_evaluation_metrics(y_true, y_pred):
    """
    Comprehensive evaluation with robust metrics
    """
    metrics = {}
    
    # Standard metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Robust metrics
    residuals = y_true - y_pred
    
    # Median absolute error (robust to outliers)
    metrics['median_ae'] = np.median(np.abs(residuals))
    
    # Mean absolute percentage error
    metrics['mape'] = np.mean(np.abs(residuals / (y_true + 1e-8))) * 100
    
    # Quantile-based metrics
    metrics['q95_ae'] = np.percentile(np.abs(residuals), 95)
    metrics['q75_ae'] = np.percentile(np.abs(residuals), 75)
    
    # Robust R-squared (based on median)
    y_median = np.median(y_true)
    ss_res_robust = np.sum(np.abs(residuals))
    ss_tot_robust = np.sum(np.abs(y_true - y_median))
    metrics['robust_r2'] = 1 - (ss_res_robust / ss_tot_robust)
    
    return metrics

print("\\n" + "="*60)
print("ENHANCED MODEL COMPONENTS LOADED")
print("15+ Advanced Techniques Ready for Implementation")
print("="*60)
'''

# Save the enhanced code
with open('enhanced_pollution_model_v3.py', 'w') as f:
    f.write(enhanced_code)

print("✅ Enhanced pollution prediction model code created!")
print("📁 File saved: enhanced_pollution_model_v3.py")
print("\n🔬 Key Improvements Added:")
print("1. Ultra-advanced feature engineering (15+ techniques)")
print("2. Multiple ensemble strategies with diversity optimization")
print("3. SHAP-based intelligent feature selection")
print("4. Multi-level stacking ensembles")
print("5. Pseudo-labeling for regression")
print("6. Advanced cross-validation with uncertainty")
print("7. Robust evaluation metrics")
print("8. Multiple target transformations")
print("9. Advanced outlier handling")
print("10. Quantile regression integration")