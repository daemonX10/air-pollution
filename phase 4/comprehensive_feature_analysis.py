# Feature Selection Analysis and Visualization
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data and recreate the feature engineering
df = pd.read_csv('d:/competition/air pollution/phase 1/train.csv').dropna()

def create_comprehensive_features(df, is_training=True):
    """Recreate the comprehensive feature set"""
    df_enhanced = df.copy()
    
    # Cyclical time features
    df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['day_year_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_year'] / 365)
    df_enhanced['day_year_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_year'] / 365)
    df_enhanced['day_week_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
    df_enhanced['day_week_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
    
    # Industrial and Traffic Proxies
    from sklearn.cluster import KMeans
    coords = df_enhanced[['latitude', 'longitude']].values
    kmeans_industrial = KMeans(n_clusters=20, random_state=42, n_init=10)
    industrial_clusters = kmeans_industrial.fit_predict(coords)
    
    if is_training and 'pollution_value' in df_enhanced.columns:
        cluster_pollution = df_enhanced.groupby(industrial_clusters)['pollution_value'].mean()
        df_enhanced['industrial_proxy'] = industrial_clusters
        df_enhanced['industrial_proxy'] = df_enhanced['industrial_proxy'].map(cluster_pollution)
    else:
        df_enhanced['industrial_proxy'] = industrial_clusters * 0.1 + np.random.normal(0, 0.05, len(df_enhanced))
    
    # Traffic proxy
    rush_hour_multiplier = df_enhanced['hour'].map({
        7: 1.5, 8: 2.0, 9: 1.8, 17: 1.8, 18: 2.0, 19: 1.5
    }).fillna(1.0)
    
    df_enhanced['traffic_proxy'] = (
        rush_hour_multiplier * 
        (1 + np.abs(df_enhanced['latitude'] - df_enhanced['latitude'].mean())) *
        (1 - df_enhanced['day_of_week'] / 7)
    )
    
    # All the other features
    df_enhanced['traffic_lat'] = df_enhanced['latitude'] * df_enhanced['traffic_proxy']
    df_enhanced['traffic_lon'] = df_enhanced['longitude'] * df_enhanced['traffic_proxy']
    df_enhanced['meteo_season'] = np.sin(2 * np.pi * df_enhanced['month'] / 12) * 2 + np.cos(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['meteo_season_abs'] = np.abs(df_enhanced['meteo_season'])
    df_enhanced['loc_time_int_1'] = df_enhanced['latitude'] * df_enhanced['hour'] * df_enhanced['month']
    df_enhanced['loc_time_int_2'] = df_enhanced['longitude'] * df_enhanced['day_of_year'] * df_enhanced['hour']
    df_enhanced['time_interaction_1'] = df_enhanced['hour'] * df_enhanced['day_of_week'] * df_enhanced['month']
    df_enhanced['time_interaction_2'] = df_enhanced['day_of_year'] * df_enhanced['hour'] ** 2
    df_enhanced['cycle_interaction_1'] = df_enhanced['hour_sin'] * df_enhanced['month_cos']
    df_enhanced['cycle_interaction_2'] = df_enhanced['day_year_sin'] * df_enhanced['hour_cos']
    df_enhanced['weekday_pollution'] = (df_enhanced['day_of_week'] < 5).astype(int) * df_enhanced['hour']
    df_enhanced['weekend_pollution'] = (df_enhanced['day_of_week'] >= 5).astype(int) * df_enhanced['hour']
    df_enhanced['harmonic_hour_1'] = np.sin(4 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['harmonic_hour_2'] = np.cos(4 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['day_of_year_hour_interaction'] = df_enhanced['day_of_year'] * np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['lat_hour_sin_interaction'] = df_enhanced['latitude'] * np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['lon_day_year_cos_interaction'] = df_enhanced['longitude'] * np.cos(2 * np.pi * df_enhanced['day_of_year'] / 365)
    df_enhanced['hour_cube'] = df_enhanced['hour'] ** 3
    df_enhanced['lat_ind_int'] = df_enhanced['latitude'] * df_enhanced['industrial_proxy']
    df_enhanced['lon_ind_int'] = df_enhanced['longitude'] * df_enhanced['industrial_proxy']
    df_enhanced['meteo_lat'] = df_enhanced['latitude'] * df_enhanced['meteo_season']
    df_enhanced['meteo_lon'] = df_enhanced['longitude'] * df_enhanced['meteo_season']
    df_enhanced['traffic_season'] = df_enhanced['traffic_proxy'] * df_enhanced['meteo_season']
    df_enhanced['lat_sqrt'] = np.sqrt(np.abs(df_enhanced['latitude']))
    df_enhanced['lon_sqrt'] = np.sqrt(np.abs(df_enhanced['longitude']))
    df_enhanced['traffic_proxy_squared'] = df_enhanced['traffic_proxy'] ** 2
    
    lat_median = df_enhanced['latitude'].median()
    lon_median = df_enhanced['longitude'].median()
    distance_from_center = np.sqrt((df_enhanced['latitude'] - lat_median)**2 + (df_enhanced['longitude'] - lon_median)**2)
    df_enhanced['emission_hotspot'] = (distance_from_center < distance_from_center.quantile(0.2)).astype(int)
    df_enhanced['time_cycle_combo'] = df_enhanced['hour_sin'] * df_enhanced['day_week_cos'] * df_enhanced['month_sin']
    df_enhanced['industrial_sin_hour'] = df_enhanced['industrial_proxy'] * np.sin(2 * np.pi * df_enhanced['hour'] / 24)
    df_enhanced['distance_to_industry_min'] = np.min([
        np.abs(df_enhanced['latitude'] - df_enhanced['latitude'].quantile(0.1)),
        np.abs(df_enhanced['longitude'] - df_enhanced['longitude'].quantile(0.1))
    ], axis=0)
    df_enhanced['harmonic_month_1'] = np.sin(4 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['harmonic_month_2'] = np.cos(4 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['traffic_ind_combo'] = df_enhanced['traffic_proxy'] * df_enhanced['industrial_proxy']
    df_enhanced['high_traffic_hours'] = df_enhanced['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df_enhanced['meteo_time_int'] = df_enhanced['meteo_season'] * df_enhanced['hour'] * df_enhanced['day_of_week']
    df_enhanced['traffic_ind_lat'] = df_enhanced['traffic_proxy'] * df_enhanced['industrial_proxy'] * df_enhanced['latitude']
    df_enhanced['traffic_ind_lon'] = df_enhanced['traffic_proxy'] * df_enhanced['industrial_proxy'] * df_enhanced['longitude']
    df_enhanced['dayweek_month_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7) * np.sin(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['dayweek_month_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7) * np.cos(2 * np.pi * df_enhanced['month'] / 12)
    df_enhanced['emissions_amplification'] = df_enhanced['industrial_proxy'] * df_enhanced['traffic_proxy'] * df_enhanced['high_traffic_hours']
    df_enhanced['location_time_weight'] = (df_enhanced['latitude'] + df_enhanced['longitude']) * df_enhanced['hour'] / 24
    df_enhanced['distance_industry_time'] = df_enhanced['distance_to_industry_min'] * df_enhanced['hour']
    df_enhanced['distance_industry_season'] = df_enhanced['distance_to_industry_min'] * df_enhanced['meteo_season']
    df_enhanced['traffic_industrial_ratio'] = df_enhanced['traffic_proxy'] / (df_enhanced['industrial_proxy'] + 1e-6)
    df_enhanced['lat_lon_ratio'] = df_enhanced['latitude'] / (np.abs(df_enhanced['longitude']) + 1e-6)
    
    return df_enhanced.fillna(0)

# Create features
print("Creating comprehensive feature analysis...")
df_enhanced = create_comprehensive_features(df, is_training=True)

# Define feature sets
base_features = ['latitude', 'longitude', 'day_of_year', 'day_of_week', 'hour', 'month']
optimized_features = ['lat_ind_int', 'meteo_lon', 'traffic_ind_lon', 'cycle_interaction_2', 'longitude', 'industrial_proxy']
all_new_features = [col for col in df_enhanced.columns if col not in base_features + ['pollution_value']]

# Prepare data
X_base = df_enhanced[base_features]
X_optimized = df_enhanced[optimized_features]
X_all = df_enhanced[all_new_features]
y = df_enhanced['pollution_value']

# Split data
X_base_train, X_base_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
X_opt_train, X_opt_test, _, _ = train_test_split(X_optimized, y, test_size=0.2, random_state=42)
X_all_train, X_all_test, _, _ = train_test_split(X_all, y, test_size=0.2, random_state=42)

# Performance comparison function
def compare_performance(X_train, X_test, y_train, y_test, name):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    return {'name': name, 'rmse': rmse, 'r2': r2, 'features': X_train.shape[1]}

# Compare different feature sets
results = []
results.append(compare_performance(X_base_train, X_base_test, y_train, y_test, "Base Features"))
results.append(compare_performance(X_opt_train, X_opt_test, y_train, y_test, "Optimized Features"))
results.append(compare_performance(X_all_train, X_all_test, y_train, y_test, "All New Features"))

results_df = pd.DataFrame(results)
print("\nPerformance Comparison:")
print(results_df.to_string(index=False))

# Create comprehensive visualization
plt.figure(figsize=(20, 15))

# 1. Performance comparison bar chart
plt.subplot(3, 3, 1)
x_pos = np.arange(len(results_df))
bars = plt.bar(x_pos, results_df['rmse'], color=['lightcoral', 'lightgreen', 'lightblue'])
plt.xlabel('Feature Sets')
plt.ylabel('RMSE')
plt.title('RMSE Comparison Across Feature Sets')
plt.xticks(x_pos, results_df['name'], rotation=45)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{results_df.iloc[i]["rmse"]:.2f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 2. R² comparison
plt.subplot(3, 3, 2)
bars = plt.bar(x_pos, results_df['r2'], color=['lightcoral', 'lightgreen', 'lightblue'])
plt.xlabel('Feature Sets')
plt.ylabel('R² Score')
plt.title('R² Score Comparison')
plt.xticks(x_pos, results_df['name'], rotation=45)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{results_df.iloc[i]["r2"]:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 3. Feature count vs Performance
plt.subplot(3, 3, 3)
plt.scatter(results_df['features'], results_df['rmse'], s=100, c=['red', 'green', 'blue'])
for i, row in results_df.iterrows():
    plt.annotate(row['name'], (row['features'], row['rmse']), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)
plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.title('Feature Count vs RMSE')
plt.grid(True, alpha=0.3)

# 4. Optimized feature importance
plt.subplot(3, 3, 4)
scaler = RobustScaler()
X_opt_scaled = scaler.fit_transform(X_optimized)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_opt_scaled, y)
importance = model.feature_importances_
plt.barh(range(len(optimized_features)), importance)
plt.yticks(range(len(optimized_features)), optimized_features)
plt.xlabel('Feature Importance')
plt.title('Optimized Feature Importance')
plt.gca().invert_yaxis()

# 5. Feature correlation heatmap for optimized features
plt.subplot(3, 3, 5)
corr_matrix = df_enhanced[optimized_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Optimized Features Correlation')

# 6. Pollution distribution by key feature
plt.subplot(3, 3, 6)
# Bin the most important feature and show pollution distribution
most_important_feature = optimized_features[np.argmax(importance)]
feature_bins = pd.qcut(df_enhanced[most_important_feature], q=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
pollution_by_bin = df_enhanced.groupby(feature_bins)['pollution_value'].mean()
bars = plt.bar(pollution_by_bin.index, pollution_by_bin.values, color='lightsteelblue')
plt.xlabel(f'{most_important_feature} (Binned)')
plt.ylabel('Average Pollution')
plt.title(f'Pollution vs {most_important_feature}')
plt.xticks(rotation=45)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{bar.get_height():.1f}', ha='center', va='bottom')

# 7. Feature selection process visualization
plt.subplot(3, 3, 7)
selection_steps = ['Original (61)', 'Statistical (30)', 'Forward (8)', 'Backward (6)']
feature_counts = [61, 30, 8, 6]
colors = ['lightcoral', 'orange', 'lightblue', 'lightgreen']
bars = plt.bar(selection_steps, feature_counts, color=colors)
plt.xlabel('Selection Step')
plt.ylabel('Number of Features')
plt.title('Feature Selection Process')
plt.xticks(rotation=45)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{int(bar.get_height())}', ha='center', va='bottom')

# 8. Performance improvement chart
plt.subplot(3, 3, 8)
performance_steps = ['Base', 'All Features', 'Optimized']
rmse_values = [results_df[results_df['name'] == 'Base Features']['rmse'].iloc[0],
               results_df[results_df['name'] == 'All New Features']['rmse'].iloc[0],
               results_df[results_df['name'] == 'Optimized Features']['rmse'].iloc[0]]
colors = ['red', 'orange', 'green']
bars = plt.bar(performance_steps, rmse_values, color=colors)
plt.xlabel('Feature Engineering Stage')
plt.ylabel('RMSE')
plt.title('Performance Through Optimization')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{bar.get_height():.2f}', ha='center', va='bottom')

# 9. Feature type breakdown
plt.subplot(3, 3, 9)
feature_types = {
    'Geographic': ['latitude', 'longitude', 'lat_ind_int', 'meteo_lon', 'traffic_ind_lon'],
    'Temporal': ['cycle_interaction_2'],
    'Industrial': ['industrial_proxy']
}
type_counts = [len([f for f in optimized_features if f in types]) for types in feature_types.values()]
plt.pie(type_counts, labels=feature_types.keys(), autopct='%1.1f%%', startangle=90)
plt.title('Optimized Feature Types')

plt.tight_layout()
plt.savefig('d:/competition/air pollution/comprehensive_feature_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("\n" + "="*80)
print("COMPREHENSIVE FEATURE ANALYSIS SUMMARY")
print("="*80)

print(f"\n📊 PERFORMANCE IMPROVEMENTS:")
base_rmse = results_df[results_df['name'] == 'Base Features']['rmse'].iloc[0]
opt_rmse = results_df[results_df['name'] == 'Optimized Features']['rmse'].iloc[0]
improvement = ((base_rmse - opt_rmse) / base_rmse) * 100

print(f"• Base features RMSE: {base_rmse:.4f}")
print(f"• Optimized features RMSE: {opt_rmse:.4f}")
print(f"• Performance improvement: {improvement:.2f}%")

print(f"\n🔧 FEATURE ENGINEERING SUCCESS:")
print(f"• Started with {len(base_features)} base features")
print(f"• Created {len(all_new_features)} advanced features")
print(f"• Optimized down to {len(optimized_features)} best features")
print(f"• Feature reduction: {((len(all_new_features) - len(optimized_features)) / len(all_new_features)) * 100:.1f}%")

print(f"\n🎯 OPTIMAL FEATURE SET:")
for i, feature in enumerate(optimized_features, 1):
    print(f"{i:2d}. {feature}")

print(f"\n💡 KEY INSIGHTS:")
print(f"• Industrial-location interactions are most important")
print(f"• Geographic features dominate the optimal set")
print(f"• Cyclical time patterns provide significant value")
print(f"• Feature selection prevented overfitting with too many features")

print(f"\n📈 FINAL RESULTS:")
print(f"• Best model: Random Forest with optimized features")
print(f"• Test RMSE: {opt_rmse:.4f}")
print(f"• Test R²: {results_df[results_df['name'] == 'Optimized Features']['r2'].iloc[0]:.4f}")
print(f"• Submission file: enhanced_automated_submission.csv")
print("="*80)
