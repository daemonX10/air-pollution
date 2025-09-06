# Air Pollution Prediction Model - Phase 4
## Advanced Feature Engineering and Automated Selection

Welcome to the Air Pollution Prediction project! This repository contains an advanced machine learning solution for predicting air pollution levels using comprehensive feature engineering and automated feature selection.

---

## 🎯 **Project Overview**

This project implements a sophisticated air pollution prediction model that:
- **Achieves 55.27% R² score** with **17.65 RMSE** on test data
- **Engineers 61 advanced features** from basic temporal and geographic data
- **Automatically selects optimal feature subset** (6 features from 61)
- **Generates competition-ready predictions** in CSV format

---

## 📋 **Prerequisites**

### Required Software
- **Python 3.8+** (tested with Python 3.12)
- **Git** (for version control)

### Required Libraries
Install the following Python packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm joblib warnings
```

### Required Data Files
You'll need access to the original competition data:
- `train.csv` - Training dataset (should be in `d:/competition/air pollution/phase 1/`)
- `test.csv` - Test dataset (for final predictions)

---

## 🚀 **Quick Start Guide**

### Step 1: Clone/Download the Repository
```bash
# If using git
git clone <repository-url>
cd "air pollution/phase 4"

# Or download and extract the files to your working directory
```

### Step 2: Verify File Structure
Ensure your directory structure looks like this:
```
phase 4/
├── README.md                                           # This file
├── enhanced_pollution_model_with_advanced_features.py  # Main model script
├── comprehensive_feature_analysis.py                   # Feature analysis script
├── FINAL_FEATURE_ENGINEERING_RESULTS.md              # Detailed results documentation
├── enhanced_automated_submission.csv                  # Final predictions
├── feature_selection_results.txt                     # Selected features summary
├── results.log                                       # Execution log
├── automated_feature_analysis.png                    # Feature importance plot
└── comprehensive_feature_analysis.png                # Feature analysis visualization
```

### Step 3: Prepare Data Paths
Update the data paths in the Python scripts to match your local setup:
- Open `enhanced_pollution_model_with_advanced_features.py`
- Update line 21: `df = pd.read_csv('d:/competition/air pollution/phase 1/train.csv')`
- Update line 550+: Test data path for final predictions

### Step 4: Run the Complete Pipeline
```bash
# Option 1: Run the comprehensive analysis (recommended for first time)
python comprehensive_feature_analysis.py

# Option 2: Run the full model training and prediction pipeline
python enhanced_pollution_model_with_advanced_features.py
```

---

## 📊 **Understanding the Process**

### Phase 1: Data Loading and Cleaning
- Loads training data and removes missing values
- Detects and caps extreme outliers using IQR method
- Preserves data integrity while handling anomalies

### Phase 2: Comprehensive Feature Engineering
The model creates **61 total features** including:

#### **Cyclical Time Features**
- Hour, month, day-of-year, day-of-week sine/cosine transformations
- Captures periodic patterns in pollution data

#### **Industrial and Traffic Proxies**
- Geographic clustering to identify industrial areas
- Rush hour traffic intensity modeling
- Location-based pollution hotspot detection

#### **Advanced Interaction Features**
- Geographic-temporal interactions (lat × hour, lon × season)
- Industrial-traffic combinations
- Meteorological-location interactions

#### **Mathematical Transformations**
- Polynomial features (squared, cubic terms)
- Square root transformations
- Harmonic decompositions

### Phase 3: Automated Feature Selection
The system uses a **multi-stage selection process**:

1. **Statistical Selection**: Uses f_regression to select top 30 features
2. **Forward Selection**: Iteratively adds features that improve model performance
3. **Backward Elimination**: Removes redundant features
4. **Cross-validation**: Validates feature combinations using time series splits

### Phase 4: Model Training and Comparison
Tests multiple algorithms:
- **Random Forest** ⭐ (Best performer)
- **LightGBM**
- **Huber Regressor**
- **Ensemble Methods**

### Phase 5: Final Predictions
Generates competition-ready CSV file with predictions.

---

## 🔧 **Final Model Architecture**

### **Optimal Feature Set (6 features)**
1. `lat_ind_int` - Latitude × Industrial proxy interaction
2. `meteo_lon` - Longitude × Meteorological season interaction  
3. `traffic_ind_lon` - Traffic × Industrial × Longitude interaction
4. `cycle_interaction_2` - Day-year sine × Hour cosine interaction
5. `longitude` - Base geographic coordinate
6. `industrial_proxy` - Industrial activity indicator

### **Model Performance**
- **Algorithm**: Random Forest (100 estimators)
- **Test RMSE**: 17.6492
- **Test R²**: 0.5527
- **Cross-validation**: Time series splits for robust validation

---

## 📈 **Results and Outputs**

### **Generated Files**
After running the pipeline, you'll get:

1. **`enhanced_automated_submission.csv`** - Final predictions ready for competition submission
2. **`feature_selection_results.txt`** - Summary of selected features and model performance
3. **`automated_feature_analysis.png`** - Feature importance visualization
4. **`comprehensive_feature_analysis.png`** - Detailed feature analysis plots
5. **`results.log`** - Complete execution log with detailed metrics

### **Performance Metrics**
- **Feature Reduction**: 90% (from 61 to 6 features)
- **Model Accuracy**: 55.27% R² score
- **Prediction Error**: 17.65 RMSE
- **Processing Time**: ~2-5 minutes on modern hardware

---

## 🛠️ **Customization Options**

### **Adjusting Feature Selection**
In `enhanced_pollution_model_with_advanced_features.py`, modify:
- `k_best=30` (line ~380) - Number of features for statistical selection
- `max_features_to_add=15` (line ~420) - Maximum features in forward selection
- Cross-validation folds and scoring metrics

### **Model Hyperparameters**
Modify model parameters around line 500+:
```python
# Random Forest parameters
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Tree depth
    random_state=42,
    n_jobs=-1
)
```

### **Adding New Features**
Extend the `create_comprehensive_features()` function (line ~45) to add your own engineered features.

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **"File not found" errors**
- Update data file paths in the scripts to match your local directory structure
- Ensure `train.csv` and `test.csv` are in the correct locations

#### **Memory errors**
- Reduce the number of features in initial selection
- Use smaller `n_estimators` for Random Forest

#### **Poor performance**
- Check data quality and preprocessing steps
- Verify feature engineering is producing expected ranges
- Consider different model parameters

#### **Slow execution**
- Reduce cross-validation folds
- Use fewer features in forward selection
- Set `n_jobs=-1` for parallel processing

### **Getting Help**
1. Check the `results.log` file for detailed execution information
2. Review the `FINAL_FEATURE_ENGINEERING_RESULTS.md` for methodology details
3. Examine feature importance plots to understand model behavior

---

## 📚 **Understanding the Results**

### **Feature Importance**
The selected features represent different aspects of air pollution:
- **Geographic interactions** capture spatial patterns
- **Time-based features** model temporal trends
- **Industrial proxies** identify pollution sources
- **Traffic interactions** account for vehicle emissions

### **Model Interpretation**
- **R² = 0.5527** means the model explains ~55% of pollution variance
- **RMSE = 17.65** indicates average prediction error in pollution units
- **Feature reduction** prevents overfitting while maintaining performance

---

## 🎉 **Success Metrics**

You've successfully replicated the model when you see:
- ✅ All 61 features created without errors
- ✅ Automated selection reduces to 6 optimal features  
- ✅ Final model achieves R² > 0.55
- ✅ `enhanced_automated_submission.csv` generated with 2740+ predictions
- ✅ Feature importance plots generated successfully

---

## 📝 **Next Steps**

### **For Competition Participants**
1. Submit `enhanced_automated_submission.csv` to the competition platform
2. Monitor leaderboard performance
3. Consider ensemble methods with other models

### **For Further Development**
1. Experiment with deep learning approaches
2. Add weather data if available
3. Try different feature engineering techniques
4. Implement online learning for real-time predictions

### **For Research**
1. Analyze feature importance for domain insights
2. Study temporal patterns in pollution data
3. Investigate geographic clustering results
4. Compare with other environmental prediction methods

---

## 📞 **Contact & Support**

For questions about this implementation:
- Review the detailed documentation in `FINAL_FEATURE_ENGINEERING_RESULTS.md`
- Check execution logs in `results.log`
- Analyze the generated visualizations for insights

---

## 📄 **License & Attribution**

This code is provided for educational and research purposes. When using this work:
- Cite the methodology if publishing results
- Acknowledge the comprehensive feature engineering approach
- Share improvements with the community

---

**Happy modeling! 🚀**

*Last updated: September 2025*
