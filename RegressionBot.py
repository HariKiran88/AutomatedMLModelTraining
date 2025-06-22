import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import os
import pickle
import tempfile
from datetime import datetime
import concurrent.futures
import threading
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold, RFE
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import pygam
import tensorflow as tf
from tensorflow import keras
from keras import layers, optimizers, callbacks
import warnings
import io
import base64
import time
import json
# Try to import Boruta, make it optional
try:
  from boruta import BorutaPy
  BORUTA_AVAILABLE = True
except ImportError:
  BORUTA_AVAILABLE = False
  print("Boruta not available. Feature selection will skip Boruta method.")
# Global variable to store training results
training_results = None
class MeanBaselineRegressor(BaseEstimator, RegressorMixin):
  def __init__(self):
    self.mean_ = None
  def fit(self, X, y):
    self.mean_ = np.mean(y)
    return self
  def predict(self, X):
    return np.full(X.shape[0], self.mean_)
class MedianBaselineRegressor(BaseEstimator, RegressorMixin):
  def __init__(self):
    self.median_ = None
  def fit(self, X, y):
    self.median_ = np.median(y)
    return self
  def predict(self, X):
    return np.full(X.shape[0], self.median_)
class BestFitNeuralNetwork(BaseEstimator, RegressorMixin):
  def __init__(self, hidden_layers=[64, 32], activation='relu', learning_rate=0.001,
         epochs=100, batch_size=32, dropout_rate=0.2, patience=10):
    self.hidden_layers = hidden_layers
    self.activation = activation
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.batch_size = batch_size
    self.dropout_rate = dropout_rate
    self.patience = patience
    self.model = None
    self.scaler = StandardScaler()
  def fit(self, X, y):
    # Scale features
    X_scaled = self.scaler.fit_transform(X)
    # Build model
    model = keras.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))
    # Add hidden layers
    for units in self.hidden_layers:
      model.add(layers.Dense(units, activation=self.activation))
      model.add(layers.Dropout(self.dropout_rate))
    # Output layer
    model.add(layers.Dense(1))
    # Compile
    model.compile(
      optimizer=optimizers.Adam(learning_rate=self.learning_rate),
      loss='mse'
    )
    # Early stopping
    early_stopping = callbacks.EarlyStopping(
      monitor='val_loss',
      patience=self.patience,
      restore_best_weights=True
    )
    # Train
    model.fit(
      X_scaled, y,
      epochs=self.epochs,
      batch_size=self.batch_size,
      validation_split=0.2,
      callbacks=[early_stopping],
      verbose=0
    )
    self.model = model
    return self
  def predict(self, X):
    X_scaled = self.scaler.transform(X)
    return self.model.predict(X_scaled).flatten()
class HPODeepNeuralNetwork(BestFitNeuralNetwork):
  def __init__(self, max_layers=5, max_units=256, **kwargs):
    super().__init__(**kwargs)
    self.max_layers = max_layers
    self.max_units = max_units
  def fit(self, X, y):
    # Simple architecture search (in a real HPO, this would use Bayesian optimization or similar)
    best_val_loss = float('inf')
    best_architecture = []
    # Try a few different architectures
    architectures = [
      [64, 32],
      [128, 64, 32],
      [256, 128, 64, 32],
      [128, 128, 128],
      [64, 64, 64, 64]
    ]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_val_scaled = self.scaler.transform(X_val)
    for architecture in architectures:
      # Build model
      model = keras.Sequential()
      model.add(layers.Input(shape=(X.shape[1],)))
      # Add hidden layers
      for units in architecture:
        model.add(layers.Dense(units, activation=self.activation))
        model.add(layers.Dropout(self.dropout_rate))
      # Output layer
      model.add(layers.Dense(1))
      # Compile
      model.compile(
        optimizer=optimizers.Adam(learning_rate=self.learning_rate),
        loss='mse'
      )
      # Train
      model.fit(
        X_train_scaled, y_train,
        epochs=min(30, self.epochs), # Quick training for architecture search
        batch_size=self.batch_size,
        verbose=0
      )
      # Evaluate
      val_loss = model.evaluate(X_val_scaled, y_val, verbose=0)
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_architecture = architecture
    # Train final model with best architecture
    self.hidden_layers = best_architecture
    return super().fit(X, y)
class GradientBoostedGAM(BaseEstimator, RegressorMixin):
  def __init__(self, n_splines=10, spline_order=3, lam=0.1, n_estimators=100):
    self.n_splines = n_splines
    self.spline_order = spline_order
    self.lam = lam
    self.n_estimators = n_estimators
    self.model = None
  def fit(self, X, y):
    try:
      # Create GAM model with splines for each feature
      gam = pygam.GAM(
        pygam.s(0, n_splines=self.n_splines, spline_order=self.spline_order, lam=self.lam) +
        sum(pygam.s(i, n_splines=self.n_splines, spline_order=self.spline_order, lam=self.lam)
          for i in range(1, X.shape[1]))
      )
      # Fit the model
      self.model = gam.fit(X, y)
      return self
    except:
      # Fallback to GradientBoostingRegressor if GAM fails
      self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, random_state=42)
      self.model.fit(X, y)
      return self
  def predict(self, X):
    return self.model.predict(X)
class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
  def __init__(self, base_models=None, weights=None):
    self.base_models = base_models or []
    self.weights = weights
    self.ensemble = None
  def fit(self, X, y):
    if not self.base_models:
      # Default base models
      self.base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
        ('lgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42))
      ]
    # Create and fit the ensemble
    self.ensemble = VotingRegressor(
      estimators=self.base_models,
      weights=self.weights
    )
    self.ensemble.fit(X, y)
    return self
  def predict(self, X):
    return self.ensemble.predict(X)
class PredictiveEnsembleRegressor(BaseEstimator, RegressorMixin):
  def __init__(self, base_models=None, meta_model=None):
    self.base_models = base_models or []
    self.meta_model = meta_model or Ridge()
    self.ensemble = None
  def fit(self, X, y):
    if not self.base_models:
      # Default base models
      self.base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
        ('lgbm', lgb.LGBMRegressor(n_estimators=100, random_state=42)),
        ('cat', CatBoostRegressor(iterations=100, random_seed=42, verbose=0))
      ]
    # Create and fit the stacking ensemble
    self.ensemble = StackingRegressor(
      estimators=self.base_models,
      final_estimator=self.meta_model,
      cv=5
    )
    self.ensemble.fit(X, y)
    return self
  def predict(self, X):
    return self.ensemble.predict(X)
class HighPerformancePredictiveEnsembleRegressor(PredictiveEnsembleRegressor):
  def __init__(self):
    # More sophisticated ensemble with more base models and a better meta-learner
    base_models = [
      ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
      ('xgb', XGBRegressor(n_estimators=200, random_state=42)),
      ('lgbm', lgb.LGBMRegressor(n_estimators=200, random_state=42)),
      ('cat', CatBoostRegressor(iterations=200, random_seed=42, verbose=0)),
      ('gb', GradientBoostingRegressor(n_estimators=200, random_state=42)),
      ('ridge', Ridge(alpha=1.0, random_state=42))
    ]
    meta_model = XGBRegressor(n_estimators=100, random_state=42)
    super().__init__(base_models=base_models, meta_model=meta_model)
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# Thread-safe logging
log_lock = threading.Lock()
# === HELPER FUNCTIONS ===
def get_img_as_base64(fig):
  buf = io.BytesIO()
  fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
  buf.seek(0)
  img_str = base64.b64encode(buf.read()).decode('utf-8')
  buf.close()
  plt.close(fig)
  return img_str
def wape(y_true, y_pred):
  return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
# === HYPERPARAMETER GRIDS ===
def get_hyperparameter_grids():
  """Define hyperparameter grids for different models - Reduced for faster execution"""
  return {
    "LightGBM": {
      'n_estimators': [300, 500, 1000], # Reduced from [100, 300, 500, 1000]
      'num_leaves': [31, 50, 100],
      'max_depth': [6, 8, 10], # Reduced from [5, 8, 10]
      'learning_rate': [0.01, 0.05, 0.1],
      'min_child_samples': [10, 20], # Reduced from [10, 20, 30]
      'subsample': [0.8, 0.9], # Reduced from [0.8, 0.9, 1.0]
      'colsample_bytree': [0.8, 0.9] # Reduced from [0.8, 0.9, 1.0]
    },
    "XGBoost": {
      'n_estimators': [300, 500, 1000], # Reduced
      'max_depth': [3, 6, 9],
      'learning_rate': [0.01, 0.05, 0.1],
      'subsample': [0.8, 0.9], # Reduced
      'colsample_bytree': [0.8, 0.9], # Reduced
      'gamma': [0, 0.1], # Reduced from [0, 0.1, 0.5]
      'min_child_weight': [1, 3] # Reduced from [1, 3, 5]
    },
    "RandomForest": {
      'n_estimators': [100, 300], # Reduced from [100, 300, 500]
      'max_depth': [5, 10, None], # Reduced from [5, 10, 15, None]
      'min_samples_split': [2, 5], # Reduced from [2, 5, 10]
      'min_samples_leaf': [1, 2], # Reduced from [1, 2, 4]
      'max_features': ['auto', 'sqrt'] # Reduced from ['auto', 'sqrt', 'log2']
    },
    "CatBoost": {
      'iterations': [300, 500, 1000], # Reduced
      'depth': [4, 6, 8], # Reduced from [4, 6, 8, 10]
      'learning_rate': [0.01, 0.05, 0.1],
      'l2_leaf_reg': [1, 3, 5], # Reduced from [1, 3, 5, 9]
      'rsm': [0.8, 0.9] # Reduced from [0.7, 0.8, 0.9, 1.0]
    }
  }
# === VISUALIZATION FUNCTIONS ===
def visualize_feature_selection(X, votes_df, selected_features):
  fig = plt.figure(figsize=(15, 10))
  # 1. Feature votes heatmap
  plt.subplot(2, 2, 1)
  if len(votes_df) > 25:
    top_features = votes_df.sort_values('Score', ascending=False).head(25).index
    votes_subset = votes_df.loc[top_features].drop('Score', axis=1)
  else:
    votes_subset = votes_df.drop('Score', axis=1)
  sns.heatmap(votes_subset, cmap='YlGnBu', cbar_kws={'label': 'Selected (1) or Not (0)'})
  plt.title('Feature Selection Methods Votes', fontsize=14)
  plt.xticks(rotation=45)
  plt.yticks(rotation=0)
  # 2. Feature importance score
  plt.subplot(2, 2, 2)
  votes_sorted = votes_df.sort_values('Score', ascending=False)
  if len(votes_sorted) > 0:
    sns.barplot(x=votes_sorted.index[:min(20, len(votes_sorted))],
         y=votes_sorted['Score'][:min(20, len(votes_sorted))])
    plt.title('Top 20 Features by Ensemble Score', fontsize=14)
    plt.xticks(rotation=90)
    threshold_score = votes_df.loc[selected_features[-1], 'Score'] if len(selected_features) > 0 else 0
    plt.axhline(y=threshold_score, color='r', linestyle='--',
         label=f'Selection Threshold (Score={threshold_score})')
    plt.legend()
  # 3. Selection method contributions
  plt.subplot(2, 2, 3)
  method_counts = votes_df.drop('Score', axis=1).sum()
  sns.barplot(x=method_counts.index, y=method_counts.values)
  plt.title('Features Selected by Each Method', fontsize=14)
  plt.ylabel('Number of Features Selected')
  plt.xticks(rotation=45)
  # 4. Selected vs Not Selected count
  plt.subplot(2, 2, 4)
  selected_count = len(selected_features)
  not_selected_count = len(votes_df) - selected_count
  sns.barplot(x=['Selected', 'Not Selected'], y=[selected_count, not_selected_count])
  plt.title(f'Feature Selection Result', fontsize=14)
  plt.ylabel('Number of Features')
  plt.tight_layout()
  return fig
def visualize_model_performance(results_df, y_test, y_pred_dict):
  fig = plt.figure(figsize=(20, 16))
  # 1. Error metrics comparison
  plt.subplot(2, 2, 1)
  metrics_df = results_df[['MAE', 'RMSE', 'WAPE']].copy()
  metrics_df = metrics_df.sort_values('WAPE')
  ax = metrics_df.plot(kind='bar', ax=plt.gca())
  plt.title('Error Metrics Comparison (Lower is Better)', fontsize=14)
  plt.ylabel('Error Value')
  plt.xticks(rotation=45)
  for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8)
  # 2. R2 comparison
  plt.subplot(2, 2, 2)
  r2_df = results_df[['R2']].copy().sort_values('R2', ascending=False)
  ax = r2_df.plot(kind='bar', ax=plt.gca(), color='green')
  plt.title('R² Score Comparison (Higher is Better)', fontsize=14)
  plt.ylabel('R² Score')
  plt.xticks(rotation=45)
  for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8)
  # 3. Best model - Actual vs Predicted
  best_model_name = results_df['WAPE'].idxmin()
  y_pred_best = y_pred_dict[best_model_name]
  plt.subplot(2, 2, 3)
  plt.scatter(y_test, y_pred_best, alpha=0.5)
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
  plt.title(f'Actual vs Predicted ({best_model_name})', fontsize=14)
  plt.xlabel('Actual Value')
  plt.ylabel('Predicted Value')
  # Show correlation
  corr = np.corrcoef(y_test, y_pred_best)[0, 1]
  plt.annotate(f'Correlation: {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
  # 4. Error distribution
  plt.subplot(2, 2, 4)
  errors = y_test - y_pred_best
  sns.histplot(errors, kde=True)
  plt.title(f'Error Distribution ({best_model_name})', fontsize=14)
  plt.xlabel('Error (Actual - Predicted)')
  # Error statistics
  mean_error = errors.mean()
  std_error = errors.std()
  plt.annotate(f'Mean Error: {mean_error:.3f}\\nStd Dev: {std_error:.3f}',
         xy=(0.05, 0.95), xycoords='axes fraction')
  plt.tight_layout()
  return fig
def visualize_feature_importance(X, best_model, model_name):
  if not hasattr(best_model, 'feature_importances_'):
    return None, "Model doesn't support feature importance visualization"
  fig = plt.figure(figsize=(12, 10))
  # Get feature importances
  importances = best_model.feature_importances_
  feature_names = X.columns
  importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
  importance_df = importance_df.sort_values('Importance', ascending=False)
  # Plot top 20 features
  top_features = importance_df.head(20)
  ax = sns.barplot(x='Importance', y='Feature', data=top_features)
  plt.title(f'Top 20 Feature Importances ({model_name})', fontsize=14)
  # Add percentage labels
  total_importance = importance_df['Importance'].sum()
  for i, v in enumerate(top_features['Importance']):
    percentage = (v / total_importance) * 100
    ax.text(v + 0.001, i, f'{percentage:.1f}%', va='center')
  plt.tight_layout()
  return fig, importance_df
# === FEATURE SELECTION WITH MULTITHREADING ===
def run_feature_selection_method(method_name, X_clean, y_clean, use_boruta=True, log_callback=None):
  """Run a single feature selection method in a separate thread"""
  try:
    if method_name == "Variance":
      if log_callback is not None:
        with log_lock:
          log_callback(f"🔄 Running Variance Threshold...")
      selector = VarianceThreshold(threshold=0.01)
      selector.fit(X_clean)
      selected = selector.get_support().astype(int)
      if log_callback is not None:
        with log_lock:
          log_callback(f"✓ Variance Threshold selected {sum(selected)} of {len(X_clean.columns)} features")
      return method_name, selected
    elif method_name == "Correlation":
      if log_callback is not None:
        with log_lock:
          log_callback(f"🔄 Running Correlation Filter...")
      corr = X_clean.corrwith(y_clean).abs()
      selected = (corr > 0.05).astype(int)
      if log_callback is not None:
        with log_lock:
          log_callback(f"✓ Correlation Filter selected {sum(selected)} of {len(X_clean.columns)} features")
      return method_name, selected
    elif method_name == "MutualInfo":
      if log_callback is not None:
        with log_lock:
          log_callback(f"🔄 Running Mutual Information...")
      mi = mutual_info_regression(X_clean, y_clean, discrete_features='auto')
      selected = (mi > np.median(mi)).astype(int)
      if log_callback is not None:
        with log_lock:
          log_callback(f"✓ Mutual Information selected {sum(selected)} of {len(X_clean.columns)} features")
      return method_name, selected
    elif method_name == "RF":
      if log_callback is not None:
        with log_lock:
          log_callback(f"🔄 Running Random Forest feature selection...")
      rf = RandomForestRegressor(n_estimators=100, random_state=42)
      rf.fit(X_clean, y_clean)
      rf_imp = rf.feature_importances_
      selected = (rf_imp > np.median(rf_imp)).astype(int)
      if log_callback is not None:
        with log_lock:
          log_callback(f"✓ Random Forest selected {sum(selected)} of {len(X_clean.columns)} features")
      return method_name, selected
    elif method_name == "XGB":
      if log_callback is not None:
        with log_lock:
          log_callback(f"🔄 Running XGBoost feature selection...")
      xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
      xgb.fit(X_clean, y_clean)
      xgb_imp = xgb.feature_importances_
      selected = (xgb_imp > np.median(xgb_imp)).astype(int)
      if log_callback is not None:
        with log_lock:
          log_callback(f"✓ XGBoost selected {sum(selected)} of {len(X_clean.columns)} features")
      return method_name, selected
    elif method_name == "LASSO":
      if log_callback is not None:
        with log_lock:
          log_callback(f"🔄 Running LASSO feature selection...")
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X_clean)
      lasso = LassoCV(cv=5, random_state=42)
      lasso.fit(X_scaled, y_clean)
      selected = (lasso.coef_ != 0).astype(int)
      if log_callback is not None:
        with log_lock:
          log_callback(f"✓ LASSO selected {sum(selected)} of {len(X_clean.columns)} features")
      return method_name, selected
    elif method_name == "RFE":
      if log_callback is not None:
        with log_lock:
          log_callback(f"🔄 Running Recursive Feature Elimination...")
      rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42),
          n_features_to_select=int(len(X_clean.columns)*0.5))
      rfe.fit(X_clean, y_clean)
      selected = rfe.support_.astype(int)
      if log_callback is not None:
        with log_lock:
          log_callback(f"✓ RFE selected {sum(selected)} of {len(X_clean.columns)} features")
      return method_name, selected
    elif method_name == "Boruta":
      if not BORUTA_AVAILABLE or not use_boruta:
        if log_callback is not None:
          with log_lock:
            log_callback(f"⏭️ Boruta skipped (not available or disabled)")
        # Return all features selected if Boruta is skipped
        return method_name, np.ones(X_clean.shape[1], dtype=int)
      if log_callback is not None:
        with log_lock:
          log_callback(f"🔄 Running Boruta (this may take a while)...")
      boruta_selector = BorutaPy(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        n_estimators='auto',
        verbose=0,
        random_state=42
      )
      boruta_selector.fit(X_clean.values, y_clean.values)
      selected = boruta_selector.support_.astype(int)
      if log_callback is not None:
        with log_lock:
          log_callback(f"✓ Boruta selected {sum(selected)} of {len(X_clean.columns)} features")
      return method_name, selected
  except Exception as e:
    if log_callback is not None:
      with log_lock:
        log_callback(f"✗ {method_name} feature selection failed: {e}")
    # Default to selecting all features if method fails
    return method_name, np.ones(X_clean.shape[1], dtype=int)
def ensemble_feature_selector(X, y, intensity=40, use_boruta=True, progress_callback=None, log_callback=None):
  """Ensemble feature selection with multiple methods using multithreading"""
  # Ensure no NaN values in X or y
  X_clean = X.copy()
  y_clean = y.copy()
  # Create votes DataFrame
  votes = pd.DataFrame(index=X_clean.columns)
  if log_callback is not None:
    log_callback("🔍 Starting Feature Selection Process...")
  # Define methods to run (conditionally include Boruta)
  methods = ["Variance", "Correlation", "MutualInfo", "RF", "XGB", "LASSO", "RFE"]
  if use_boruta and BORUTA_AVAILABLE:
    methods.append("Boruta")
    if log_callback is not None:
      log_callback("📊 Including Boruta in feature selection (this will increase time)")
  else:
    if log_callback is not None:
      log_callback("⏭️ Skipping Boruta for faster execution")
  # Run all methods in parallel
  with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(methods), os.cpu_count() or 4)) as executor:
    # Submit all tasks
    if log_callback is not None:
      log_callback(f"📊 Running {len(methods)} feature selection methods in parallel...")
    futures = {executor.submit(run_feature_selection_method, method, X_clean, y_clean, use_boruta, log_callback): method
         for method in methods}
    # Process results as they complete
    completed = 0
    for future in concurrent.futures.as_completed(futures):
      completed += 1
      try:
        method_name, selected = future.result()
        votes[method_name] = selected
        # Update progress
        try:
          if progress_callback is not None:
            progress_callback(completed/len(methods), desc=f"Feature selection: {method_name} complete")
        except Exception as e:
          pass # Silently continue if progress update fails
      except Exception as e:
        method = futures[future]
        if log_callback is not None:
          with log_lock:
            log_callback(f"✗ {method} feature selection failed with error: {e}")
        # Default to selecting all features
        votes[method] = 1
  # Final score: sum of all method votes
  votes['Score'] = votes.sum(axis=1)
  n_keep = max(1, int(len(votes) * (1 - intensity / 100)))
  top_features = votes.sort_values('Score', ascending=False).head(n_keep).index.tolist()
  if log_callback is not None:
    log_callback(f"✅ Feature selection complete! Selected {len(top_features)} of {len(X_clean.columns)} features")
    top_3 = votes.sort_values('Score', ascending=False).head(3).index.tolist()
    log_callback(f"🏆 Top 3 features: {', '.join(top_3)}")
  return top_features, votes
# === DATA PREPROCESSING ===
def preprocess_data(df, target_col, ignore_features=None, log_callback=None):
  """Preprocess dataset for modeling with improved NaN handling"""
  if ignore_features is None:
    ignore_features = []
  if log_callback is not None:
    log_callback(f"🔍 Starting data preprocessing...")
    log_callback(f"• Initial dataset shape: {df.shape}")
  # Make a copy to avoid modifying original
  df_copy = df.copy()
  # Apply specific filtering as in the successful version
  if target_col == 'ProfitMarginPercentage':
    before_filter = df_copy.shape[0]
    df_copy = df_copy[(df_copy[target_col] > -100) & (df_copy[target_col] < 150)]
    if log_callback is not None:
      log_callback(f"• Filtered target outliers: removed {before_filter - df_copy.shape[0]} rows")
  # Drop specific columns that were in the successful version
  drop_cols = ['NextIntervalDate', 'DaysInInterval']
  dropped = [col for col in drop_cols if col in df_copy.columns]
  df_copy = df_copy.drop(columns=dropped, errors='ignore')
  if dropped and log_callback is not None:
    log_callback(f"• Dropped system columns: {', '.join(dropped)}")
  # Basic info
  if log_callback is not None:
    na_cols = df_copy.columns[df_copy.isna().any()].tolist()
    if na_cols:
      log_callback(f"• Found {len(na_cols)} columns with missing values")
    else:
      log_callback(f"• No columns with missing values")
  # Drop specified columns
  ignore_in_df = [col for col in ignore_features if col in df_copy.columns]
  df_copy = df_copy.drop(columns=ignore_in_df)
  if ignore_in_df and log_callback is not None:
    log_callback(f"• Dropped user-specified columns: {', '.join(ignore_in_df)}")
  # Check if target exists
  if target_col not in df_copy.columns:
    if log_callback is not None:
      log_callback(f"❌ Error: Target column '{target_col}' not found in dataset")
    return None, None, None, f"Target column '{target_col}' not found in dataset"
  # Identify potential outliers in target
  target_outliers = 0
  if np.issubdtype(df_copy[target_col].dtype, np.number):
    q1, q3 = df_copy[target_col].quantile(0.01), df_copy[target_col].quantile(0.99)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    target_outliers = df_copy[(df_copy[target_col] < lower) | (df_copy[target_col] > upper)].shape[0]
    if log_callback is not None and target_outliers:
      log_callback(f"• Target has {target_outliers} potential outliers ({target_outliers/len(df_copy)*100:.1f}%)")
  # Handle missing values in parallel for large datasets
  def process_numeric_columns():
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
      missing = df_copy[col].isna().sum()
      if missing > 0:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
        if log_callback is not None:
          with log_lock:
            log_callback(f"• Filled {missing} missing values in '{col}' with median")
  def process_categorical_columns():
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
      missing = df_copy[col].isna().sum()
      if missing > 0:
        df_copy[col] = df_copy[col].fillna("Unknown")
        if log_callback is not None:
          with log_lock:
            log_callback(f"• Filled {missing} missing values in '{col}' with 'Unknown'")
  # Process columns in parallel for large datasets
  if df_copy.shape[0] * df_copy.shape[1] > 1000000: # Only use threading for large datasets
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      executor.submit(process_numeric_columns)
      executor.submit(process_categorical_columns)
  else:
    process_numeric_columns()
    process_categorical_columns()
  # Create X and y
  X = df_copy.drop(columns=[target_col])
  y = df_copy[target_col]
  # Check for remaining NaN values
  if X.isna().any().any():
    if log_callback is not None:
      log_callback(f"⚠️ Warning: Still have NaN values after imputation. Dropping rows with NaN.")
    # Get indices of rows with NaN
    na_indices = X[X.isna().any(axis=1)].index
    # Drop these rows from both X and y
    X = X.drop(index=na_indices)
    y = y.drop(index=na_indices)
    if log_callback is not None:
      log_callback(f"• Dropped {len(na_indices)} rows with remaining NaN values")
  # Label encode categoricals (can be done in parallel for many columns)
  categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
  le_dict = {}
  if len(categorical_features) > 10: # Only use threading for many categorical features
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(categorical_features), os.cpu_count() or 4)) as executor:
      futures = {}
      # Define function to encode a single column
      def encode_column(col):
        le = LabelEncoder()
        encoded = le.fit_transform(X[col].astype(str))
        return col, encoded, le
      # Submit encoding tasks
      for col in categorical_features:
        futures[executor.submit(encode_column, col)] = col
      # Collect results
      for future in concurrent.futures.as_completed(futures):
        col, encoded_values, le = future.result()
        X[col] = encoded_values
        le_dict[col] = le
  else:
    # For fewer columns, do sequentially
    for col in categorical_features:
      le = LabelEncoder()
      X[col] = le.fit_transform(X[col].astype(str))
      le_dict[col] = le
  if categorical_features and log_callback is not None:
    log_callback(f"• Encoded {len(categorical_features)} categorical features")
  # Final check for NaN
  if X.isna().any().any() or y.isna().any():
    if log_callback is not None:
      log_callback(f"⚠️ Warning: Still have NaN values after all processing! Last-resort cleanup.")
    # Last resort: drop any remaining NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    before = len(X)
    X = X[mask]
    y = y[mask]
    if log_callback is not None:
      log_callback(f"• Final cleanup: dropped {before - len(X)} rows with NaN values")
  # Generate summary stats
  stats = {
    "rows": df_copy.shape[0],
    "columns": df_copy.shape[1],
    "target_mean": y.mean(),
    "target_std": y.std(),
    "target_min": y.min(),
    "target_max": y.max(),
    "target_missing": df_copy[target_col].isnull().sum(),
    "target_outliers": target_outliers,
    "categorical_features": len(categorical_features),
    "numerical_features": len(X.columns) - len(categorical_features)
  }
  if log_callback is not None:
    log_callback(f"✅ Preprocessing complete! Final dataset: X={X.shape}, y={y.shape}")
  return X, y, stats, None
# === HYPERPARAMETER TUNING FUNCTIONS ===
def perform_hyperparameter_tuning(model, param_grid, X_train, y_train, search_type="grid", n_iter=20, cv=3, scoring='neg_mean_absolute_error', log_callback=None):
  """Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV - Optimized for speed"""
  try:
    if search_type == "grid":
      if log_callback:
        log_callback(f"🔧 Starting GridSearchCV with {cv}-fold CV...")
      search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
      )
    else: # randomized
      if log_callback:
        log_callback(f"🔧 Starting RandomizedSearchCV with {n_iter} iterations and {cv}-fold CV...")
      search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter, # Reduced from 50 to 20
        cv=cv, # Reduced from 5 to 3
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        random_state=42
      )
    # Perform the search
    search.fit(X_train, y_train)
    if log_callback:
      log_callback(f"✓ Hyperparameter tuning complete - Best score: {-search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_
  except Exception as e:
    if log_callback:
      log_callback(f"✗ Hyperparameter tuning failed: {e}")
    return model, {}, None
# === MODEL TRAINING WITH MULTITHREADING AND HYPERPARAMETER TUNING ===
def train_model(name, model, X_train, y_train, X_test, mode, selected_features, X_train_full, X_test_full,
        use_hp_tuning=False, hp_tuning_type="grid", log_callback=None):
  """Train a single model in a separate thread with optional hyperparameter tuning"""
  try:
    start_time = time.time()
    feature_count = len(selected_features) if mode == 'With Feature Selection' else X_train_full.shape[1]
    if log_callback is not None:
      with log_lock:
        tuning_info = f" (with {hp_tuning_type} search)" if use_hp_tuning else ""
        log_callback(f"🔄 Training {name} ({mode}){tuning_info} with {feature_count} features...")
    # Select features if needed
    X_train_use = X_train_full[selected_features] if mode == 'With Feature Selection' else X_train_full
    X_test_use = X_test_full[selected_features] if mode == 'With Feature Selection' else X_test_full
    model_key = f"{name} ({mode})"
    # Perform hyperparameter tuning if requested
    if use_hp_tuning and name in ["LightGBM", "XGBoost", "RandomForest", "CatBoost"]:
      param_grids = get_hyperparameter_grids()
      if name in param_grids:
        if log_callback is not None:
          with log_lock:
            log_callback(f"⚙️ Running hyperparameter optimization for {name}...")
        tuned_model, best_params, best_score = perform_hyperparameter_tuning(
          model, param_grids[name], X_train_use, y_train,
          search_type=hp_tuning_type, log_callback=log_callback
        )
        model = tuned_model
        model_key += f" (HP-{hp_tuning_type})"
        if log_callback is not None:
          with log_lock:
            log_callback(f"🎯 Best params for {name}: {best_params}")
    # Train the model
    if log_callback is not None:
      with log_lock:
        log_callback(f"🏋️ Training {model_key} on {X_train_use.shape[0]} samples...")
    model.fit(X_train_use, y_train)
    y_pred = model.predict(X_test_use)
    # Calculate metrics
    mae = mean_absolute_error(X_test, y_pred)
    mape = mean_absolute_percentage_error(X_test, y_pred)
    wape_val = wape(X_test, y_pred)
    rmse = np.sqrt(mean_squared_error(X_test, y_pred))
    r2 = r2_score(X_test, y_pred)
    result = {
      "Model": model_key,
      "MAE": mae,
      "MAPE": mape,
      "WAPE": wape_val,
      "RMSE": rmse,
      "R2": r2,
      "y_pred": y_pred,
      "model": model,
      "selected_features": selected_features if mode == 'With Feature Selection' else X_train_full.columns.tolist()
    }
    if log_callback is not None:
      with log_lock:
        training_time = time.time() - start_time
        log_callback(f"✅ {model_key} completed in {training_time:.1f}s - WAPE: {wape_val:.4f}, R²: {r2:.4f}")
    return result
  except Exception as e:
    if log_callback is not None:
      with log_lock:
        log_callback(f"❌ {name} ({mode}) training failed: {e}")
    return None
def train_and_evaluate_models(X, y, feature_selection_intensity=40, model_params=None,
               selected_models=None, use_hyperparameter_tuning=False, hp_tuning_type="grid",
               use_boruta=True, progress_callback=None, log_callback=None):
  """Train models and evaluate performance with configurable parameters using multithreading"""
  if log_callback is not None:
    log_callback(f"🚀 Starting model training pipeline")
    log_callback(f"📊 Dataset shape: X={X.shape}, y={y.shape}")
    log_callback(f"🎛️ Feature selection intensity: {feature_selection_intensity}%")
    log_callback(f"🤖 Selected models: {', '.join(selected_models) if selected_models else 'All models'}")
    if use_hyperparameter_tuning:
      log_callback(f"⚙️ Hyperparameter tuning: {hp_tuning_type} search enabled")
    log_callback(f"⚡ Using up to {min(6, os.cpu_count() or 4)} worker threads")
  # Use default parameters if none provided
  if model_params is None:
    model_params = {
      "lgbm": {
        "n_estimators": 2000,
        "num_leaves": 64,
        "max_depth": 8,
        "learning_rate": 0.03,
        "colsample_bytree": 0.9,
        "subsample": 0.8,
        "subsample_freq": 5,
        "min_child_samples": 20
      },
      "xgb": {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.9
      },
      "rf": {
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 5
      },
      "catboost": {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3,
        "rsm": 0.8,
        "bagging_temperature": 1
      },
      "gam": {
        "n_splines": 10,
        "spline_order": 3,
        "lam": 0.1,
        "n_estimators": 100
      },
      "nn": {
        "hidden_layers": [64, 32],
        "activation": "relu",
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "dropout_rate": 0.2
      },
      "hpo_dnn": {
        "max_layers": 5,
        "max_units": 256,
        "epochs": 150,
        "learning_rate": 0.001
      },
      "ensemble": {
        "use_meta_learner": True,
        "include_baselines": False
      }
    }
  # Track overall progress (30% feature selection, 60% model training, 10% finalization)
  overall_progress = 0
  # Feature selection (30% of progress)
  try:
    if progress_callback is not None:
      progress_callback(overall_progress, desc="Starting feature selection...")
  except Exception as e:
    pass # Silently continue if progress update fails
  # Use custom progress callback for feature selection
  def feature_selection_progress(progress, desc):
    nonlocal overall_progress
    # Scale to 0-30% range
    overall_progress = progress * 0.3
    try:
      if progress_callback is not None:
        progress_callback(overall_progress, desc=desc)
    except Exception as e:
      pass # Silently continue if progress update fails
  selected_features, votes_df = ensemble_feature_selector(
    X, y, intensity=feature_selection_intensity, use_boruta=use_boruta,
    progress_callback=feature_selection_progress,
    log_callback=log_callback
  )
  overall_progress = 0.3 # Feature selection complete
  # Split dataset
  X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
  if log_callback is not None:
    log_callback(f"📊 Train-test split: X_train={X_train_full.shape}, X_test={X_test_full.shape}")
  # Define all available models
  all_models = {
    "LightGBM": lgb.LGBMRegressor(
      objective='regression',
      num_leaves=model_params["lgbm"]["num_leaves"],
      max_depth=model_params["lgbm"]["max_depth"],
      learning_rate=model_params["lgbm"]["learning_rate"],
      colsample_bytree=model_params["lgbm"]["colsample_bytree"],
      subsample=model_params["lgbm"]["subsample"],
      subsample_freq=model_params["lgbm"]["subsample_freq"],
      min_child_samples=model_params["lgbm"]["min_child_samples"],
      n_estimators=model_params["lgbm"]["n_estimators"],
      random_state=42,
      verbosity=-1
    ),
    "XGBoost": XGBRegressor(
      objective='reg:squarederror',
      max_depth=model_params["xgb"]["max_depth"],
      learning_rate=model_params["xgb"]["learning_rate"],
      n_estimators=model_params["xgb"]["n_estimators"],
      subsample=model_params["xgb"]["subsample"],
      colsample_bytree=model_params["xgb"]["colsample_bytree"],
      random_state=42,
      verbosity=0
    ),
    "RandomForest": RandomForestRegressor(
      n_estimators=model_params["rf"]["n_estimators"],
      max_depth=model_params["rf"]["max_depth"],
      min_samples_leaf=model_params["rf"]["min_samples_leaf"],
      random_state=42,
      n_jobs=1
    ),
    "CatBoost": CatBoostRegressor(
      iterations=model_params["catboost"]["iterations"],
      depth=model_params["catboost"]["depth"],
      learning_rate=model_params["catboost"]["learning_rate"],
      l2_leaf_reg=model_params["catboost"]["l2_leaf_reg"],
      rsm=model_params["catboost"]["rsm"],
      bagging_temperature=model_params["catboost"]["bagging_temperature"],
      random_seed=42,
      verbose=0
    ),
    "GradientBoostedGAM": GradientBoostedGAM(
      n_splines=model_params["gam"]["n_splines"],
      spline_order=model_params["gam"]["spline_order"],
      lam=model_params["gam"]["lam"],
      n_estimators=model_params["gam"]["n_estimators"]
    ),
    "BestFitNN": BestFitNeuralNetwork(
      hidden_layers=model_params["nn"]["hidden_layers"],
      activation=model_params["nn"]["activation"],
      learning_rate=model_params["nn"]["learning_rate"],
      epochs=model_params["nn"]["epochs"],
      batch_size=model_params["nn"]["batch_size"],
      dropout_rate=model_params["nn"]["dropout_rate"]
    ),
    "HPO_DNN": HPODeepNeuralNetwork(
      max_layers=model_params["hpo_dnn"]["max_layers"],
      max_units=model_params["hpo_dnn"]["max_units"],
      epochs=model_params["hpo_dnn"]["epochs"],
      learning_rate=model_params["hpo_dnn"]["learning_rate"]
    ),
    "WeightedEnsemble": WeightedEnsembleRegressor(),
    "PredictiveEnsemble": PredictiveEnsembleRegressor(),
    "HighPerformanceEnsemble": HighPerformancePredictiveEnsembleRegressor(),
    "MeanBaseline": MeanBaselineRegressor(),
    "MedianBaseline": MedianBaselineRegressor()
  }
  # Filter models based on selection
  if selected_models:
    models = {name: model for name, model in all_models.items() if name in selected_models}
    if log_callback is not None:
      log_callback(f"🎯 Training only selected models: {', '.join(models.keys())}")
  else:
    models = all_models
    if log_callback is not None:
      log_callback(f"🎯 Training all {len(models)} models")
  # Train models in parallel
  training_tasks = []
  for name, model in models.items():
    for mode in ['Full Features', 'With Feature Selection']:
      training_tasks.append((name, model, y_test, y_train, mode, selected_features))
  # Parallel training
  results = []
  y_pred_dict = {}
  trained_models = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 4)) as executor:
    if log_callback is not None:
      log_callback(f"⚡ Starting parallel training of {len(training_tasks)} model configurations...")
    futures = {}
    for i, (name, model, y_test, y_train, mode, selected_features) in enumerate(training_tasks):
      # We need to create a new instance for each thread to avoid issues
      if name == "LightGBM":
        model_copy = lgb.LGBMRegressor(
          objective='regression',
          num_leaves=model_params["lgbm"]["num_leaves"],
          max_depth=model_params["lgbm"]["max_depth"],
          learning_rate=model_params["lgbm"]["learning_rate"],
          colsample_bytree=model_params["lgbm"]["colsample_bytree"],
          subsample=model_params["lgbm"]["subsample"],
          subsample_freq=model_params["lgbm"]["subsample_freq"],
          min_child_samples=model_params["lgbm"]["min_child_samples"],
          n_estimators=model_params["lgbm"]["n_estimators"],
          random_state=42,
          verbosity=-1
        )
      elif name == "XGBoost":
        model_copy = XGBRegressor(
          objective='reg:squarederror',
          max_depth=model_params["xgb"]["max_depth"],
          learning_rate=model_params["xgb"]["learning_rate"],
          n_estimators=model_params["xgb"]["n_estimators"],
          subsample=model_params["xgb"]["subsample"],
          colsample_bytree=model_params["xgb"]["colsample_bytree"],
          random_state=42,
          verbosity=0
        )
      elif name == "RandomForest":
        model_copy = RandomForestRegressor(
          n_estimators=model_params["rf"]["n_estimators"],
          max_depth=model_params["rf"]["max_depth"],
          min_samples_leaf=model_params["rf"]["min_samples_leaf"],
          random_state=42,
          n_jobs=1
        )
      elif name == "CatBoost":
        model_copy = CatBoostRegressor(
          iterations=model_params["catboost"]["iterations"],
          depth=model_params["catboost"]["depth"],
          learning_rate=model_params["catboost"]["learning_rate"],
          l2_leaf_reg=model_params["catboost"]["l2_leaf_reg"],
          rsm=model_params["catboost"]["rsm"],
          bagging_temperature=model_params["catboost"]["bagging_temperature"],
          random_seed=42,
          verbose=0
        )
      elif name == "GradientBoostedGAM":
        model_copy = GradientBoostedGAM(
          n_splines=model_params["gam"]["n_splines"],
          spline_order=model_params["gam"]["spline_order"],
          lam=model_params["gam"]["lam"],
          n_estimators=model_params["gam"]["n_estimators"]
        )
      elif name == "BestFitNN":
        model_copy = BestFitNeuralNetwork(
          hidden_layers=model_params["nn"]["hidden_layers"],
          activation=model_params["nn"]["activation"],
          learning_rate=model_params["nn"]["learning_rate"],
          epochs=model_params["nn"]["epochs"],
          batch_size=model_params["nn"]["batch_size"],
          dropout_rate=model_params["nn"]["dropout_rate"]
        )
      elif name == "HPO_DNN":
        model_copy = HPODeepNeuralNetwork(
          max_layers=model_params["hpo_dnn"]["max_layers"],
          max_units=model_params["hpo_dnn"]["max_units"],
          epochs=model_params["hpo_dnn"]["epochs"],
          learning_rate=model_params["hpo_dnn"]["learning_rate"]
        )
      elif name == "WeightedEnsemble":
        model_copy = WeightedEnsembleRegressor()
      elif name == "PredictiveEnsemble":
        model_copy = PredictiveEnsembleRegressor()
      elif name == "HighPerformanceEnsemble":
        model_copy = HighPerformancePredictiveEnsembleRegressor()
      elif name == "MeanBaseline":
        model_copy = MeanBaselineRegressor()
      elif name == "MedianBaseline":
        model_copy = MedianBaselineRegressor()
      future = executor.submit(
        train_model,
        name, model_copy, y_train, y_train, y_test, mode,
        selected_features, X_train_full, X_test_full,
        use_hyperparameter_tuning, hp_tuning_type, log_callback
      )
      futures[future] = (name, mode, i)
    # Process results as they complete
    completed = 0
    for future in concurrent.futures.as_completed(futures):
      completed += 1
      name, mode, task_index = futures[future]
      # Update progress (scale to 30-90% range)
      try:
        if progress_callback is not None:
          progress = 0.3 + (0.6 * (completed / len(futures)))
          progress_callback(progress, desc=f"Training: {name} ({mode}) complete")
      except Exception as e:
        pass
      try:
        result = future.result()
        if result is not None:
          # Extract result components
          model_key = result["Model"]
          y_pred = result["y_pred"]
          model = result["model"]
          selected_features_result = result["selected_features"]
          # Store predictions and model
          y_pred_dict[model_key] = y_pred
          trained_models[model_key] = {
            'model': model,
            'selected_features': selected_features_result
          }
          # Remove these items before adding to results list
          del result["y_pred"]
          del result["model"]
          del result["selected_features"]
          results.append(result)
      except Exception as e:
        if log_callback is not None:
          with log_lock:
            log_callback(f"❌ Error processing results for {name} ({mode}): {e}")
  # Final 10% of progress - creating results
  try:
    if progress_callback is not None:
      progress_callback(0.9, desc="Finalizing results...")
  except Exception as e:
    pass # Silently continue if progress update fails
  # Create results dataframe
  if not results:
    if log_callback is not None:
      log_callback("❌ No models completed training successfully")
    return {
      'results_df': pd.DataFrame(),
      'y_test': y_test,
      'y_pred_dict': {},
      'selected_features': selected_features,
      'votes_df': votes_df,
      'best_model_name': "None",
      'best_model': None,
      'X_for_importance': X_test_full,
      'trained_models': {}
    }
  results_df = pd.DataFrame(results).set_index("Model")
  # Get best model - FIX THE PARSING ERROR
  best_model_name = results_df['WAPE'].idxmin()
  # Handle model names that might have multiple parentheses
  # e.g., "XGBoost (With Feature Selection) (HP-random)"
  try:
    # Split on first occurrence of ' ('
    if ' (' in best_model_name:
      parts = best_model_name.split(' (', 1) # Split only on first occurrence
      model_type = parts[0].strip()
      # The rest could be "With Feature Selection) (HP-random)"
      # We need to extract the feature mode properly
      remainder = parts[1]
      if ') (' in remainder:
        # Multiple parentheses - extract the first part
        feature_mode = remainder.split(') (')[0]
      else:
        # Single parentheses
        feature_mode = remainder.rstrip(')')
    else:
      # No parentheses at all
      model_type = best_model_name
      feature_mode = 'Full Features' # Default
  except Exception as e:
    # Fallback in case of parsing issues
    model_type = best_model_name.split()[0] if ' ' in best_model_name else best_model_name
    feature_mode = 'With Feature Selection' if 'With Feature Selection' in best_model_name else 'Full Features'
  if log_callback is not None:
    log_callback(f"📊 Best model parsing: '{best_model_name}' -> Type: '{model_type}', Mode: '{feature_mode}'")
  # Get feature importance for best model
  best_model = trained_models[best_model_name]['model']
  X_for_importance = X_test_full[selected_features] if 'With Feature Selection' in best_model_name else X_test_full
  if log_callback is not None:
    log_callback(f"🏆 Best model: {best_model_name}")
    log_callback(f"📊 WAPE: {results_df.loc[best_model_name, 'WAPE']:.4f}")
    log_callback(f"📊 R²: {results_df.loc[best_model_name, 'R2']:.4f}")
    log_callback(f"✅ Model training pipeline complete!")
  try:
    if progress_callback is not None:
      progress_callback(1.0, desc="Analysis complete!")
  except Exception as e:
    pass # Silently continue if progress update fails
  # Return all results
  return {
    'results_df': results_df,
    'y_test': y_test,
    'y_pred_dict': y_pred_dict,
    'selected_features': selected_features,
    'votes_df': votes_df,
    'best_model_name': best_model_name,
    'best_model': best_model,
    'X_for_importance': X_for_importance,
    'trained_models': trained_models
  }
# === FUNCTIONS FOR PREDICTION ===
def load_model(model_path):
  """Load a saved model from pickle file"""
  try:
    with open(model_path, 'rb') as f:
      model_data = pickle.load(f)
    return model_data, None
  except Exception as e:
    return None, f"Error loading model: {str(e)}"
def preprocess_test_data(df, model_data, log_callback=None):
  """Preprocess test data for prediction"""
  try:
    required_features = model_data['selected_features']
    # Check if all required features exist
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
      return None, f"Test data is missing required features: {', '.join(missing_features)}"
    # Basic preprocessing similar to training
    df_copy = df.copy()
    # Handle missing values
    for col in required_features:
      if df_copy[col].isna().sum() > 0:
        if np.issubdtype(df_copy[col].dtype, np.number):
          # For numeric columns, fill with median
          df_copy[col] = df_copy[col].fillna(df_copy[col].median())
          if log_callback:
            log_callback(f"• Filled missing values in '{col}' with median")
        else:
          # For categorical columns, fill with "Unknown"
          df_copy[col] = df_copy[col].fillna("Unknown")
          if log_callback:
            log_callback(f"• Filled missing values in '{col}' with 'Unknown'")
    # Handle categorical features (convert to numeric)
    for col in required_features:
      if df_copy[col].dtype == 'object' or df_copy[col].dtype.name == 'category':
        # For simplicity, just use label encoding here
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        if log_callback:
          log_callback(f"• Encoded categorical feature: '{col}'")
    # Return only the required features
    return df_copy[required_features], None
  except Exception as e:
    return None, f"Error preprocessing test data: {str(e)}"
def predict_with_model(model_data, test_data, log_callback=None):
  """Make predictions using the loaded model"""
  try:
    # Extract model components
    model = model_data['model']
    selected_features = model_data['selected_features']
    model_name = model_data['name']
    if log_callback:
      log_callback(f"• Making predictions with model: {model_name}")
      log_callback(f"• Using {len(selected_features)} features")
    # Preprocess test data
    X_test, preprocess_error = preprocess_test_data(test_data, model_data, log_callback)
    if preprocess_error:
      return None, preprocess_error
    # Make predictions
    predictions = model.predict(X_test)
    # Add predictions to test data
    result_df = test_data.copy()
    result_df['Prediction'] = predictions
    if log_callback:
      log_callback(f"✅ Generated predictions for {len(test_data)} records")
      log_callback(f"• Prediction summary: Min={predictions.min():.4f}, Max={predictions.max():.4f}, Mean={predictions.mean():.4f}")
    return result_df, None
  except Exception as e:
    import traceback
    error_details = traceback.format_exc()
    return None, f"Error making predictions: {str(e)}\\n{error_details}"
# === FUNCTIONS FOR MODEL EXPORT/IMPORT ===
def export_selected_model(model_name):
  """Export the selected model to a pickle file"""
  global training_results
  if not training_results or model_name not in training_results.get('trained_models', {}):
    return None, "Model not found in results"
  try:
    # Get model info
    model_info = training_results['trained_models'][model_name]
    # Create export package
    export_data = {
      'model': model_info['model'],
      'selected_features': model_info['selected_features'],
      'name': model_name,
      'export_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      'metrics': {
        'MAE': training_results['results_df'].loc[model_name, 'MAE'],
        'WAPE': training_results['results_df'].loc[model_name, 'WAPE'],
        'RMSE': training_results['results_df'].loc[model_name, 'RMSE'],
        'R2': training_results['results_df'].loc[model_name, 'R2']
      }
    }
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = tempfile.mkdtemp()
    model_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
      pickle.dump(export_data, f)
    return model_path, None
  except Exception as e:
    import traceback
    error_details = traceback.format_exc()
    return None, f"Error exporting model: {str(e)}\\n{error_details}"
def export_analysis_schema(feature_selection_intensity, target_col, ignore_cols, model_params):
  """Export the entire analysis configuration as a schema file"""
  global training_results
  if not training_results:
    return None, "No analysis results to export"
  try:
    # Create schema object
    schema = {
      'version': '1.0',
      'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      'configuration': {
        'target_column': target_col,
        'ignored_columns': ignore_cols,
        'feature_selection_intensity': feature_selection_intensity,
        'model_parameters': model_params
      },
      'results': {
        'best_model': training_results['best_model_name'],
        'selected_features': training_results['selected_features'],
        'metrics': training_results['results_df'].to_dict(),
      }
    }
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = tempfile.mkdtemp()
    schema_path = os.path.join(output_dir, f"analysis_schema_{timestamp}.json")
    with open(schema_path, 'w') as f:
      json.dump(schema, f, indent=2)
    return schema_path, None
  except Exception as e:
    import traceback
    error_details = traceback.format_exc()
    return None, f"Error exporting schema: {str(e)}\\n{error_details}"
def import_analysis_schema(schema_file):
  """Import an analysis schema file"""
  if not schema_file:
    return None, "No schema file provided"
  try:
    # Load schema from file
    with open(schema_file.name, 'r') as f:
      schema = json.load(f)
    # Validate schema
    if 'version' not in schema or 'configuration' not in schema:
      return None, "Invalid schema file format"
    # Extract configuration
    config = schema['configuration']
    target_col = config.get('target_column')
    ignore_cols = config.get('ignored_columns', [])
    feature_selection_intensity = config.get('feature_selection_intensity', 40)
    model_params = config.get('model_parameters', {})
    return {
      'target_col': target_col,
      'ignore_cols': ignore_cols,
      'feature_selection_intensity': feature_selection_intensity,
      'model_params': model_params
    }, None
  except Exception as e:
    import traceback
    error_details = traceback.format_exc()
    return None, f"Error importing schema: {str(e)}\\n{error_details}"
def get_model_details(model_name):
  """Get details of a specific model"""
  global training_results
  if not training_results or model_name not in training_results.get('y_pred_dict', {}):
    return "No model selected", None
  # Get model details
  results_df = training_results.get('results_df', pd.DataFrame())
  if model_name not in results_df.index:
    return f"Model {model_name} not found in results", None
  # Extract metrics
  mae = results_df.loc[model_name, 'MAE']
  wape = results_df.loc[model_name, 'WAPE']
  rmse = results_df.loc[model_name, 'RMSE']
  r2 = results_df.loc[model_name, 'R2']
  # Get feature information
  model_info = training_results['trained_models'][model_name]
  feature_count = len(model_info['selected_features'])
  # Create markdown summary
  summary = f"""
  ### Model: {model_name}
  **Performance Metrics:**
  - MAE: {mae:.4f}
  - WAPE: {wape:.4f}
  - RMSE: {rmse:.4f}
  - R²: {r2:.4f}
  **Features:**
  - Using {feature_count} features
  - Top 3: {', '.join(model_info['selected_features'][:3])}
  You can export this model using the button below.
  """
  # Create model visualization - actual vs predicted scatter plot
  y_test = training_results['y_test']
  y_pred = training_results['y_pred_dict'][model_name]
  fig = plt.figure(figsize=(10, 6))
  plt.scatter(y_test, y_pred, alpha=0.5)
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
  plt.title(f'Actual vs Predicted ({model_name})', fontsize=14)
  plt.xlabel('Actual Value')
  plt.ylabel('Predicted Value')
  # Show correlation and error statistics
  corr = np.corrcoef(y_test, y_pred)[0, 1]
  errors = y_test - y_pred
  mean_error = errors.mean()
  std_error = errors.std()
  plt.annotate(f'Correlation: {corr:.3f}\\nMean Error: {mean_error:.3f}\\nStd Dev: {std_error:.3f}',
         xy=(0.05, 0.95), xycoords='axes fraction', va='top')
  plt.tight_layout()
  return summary, fig
# NEW: Function to generate prediction code
def generate_prediction_code(model_file):
  """Generate Python code for making predictions"""
  if not model_file:
    return "# Upload a model first to see the prediction code\\n# The code will show you exactly how to use your trained model outside this dashboard"
  try:
    # Load model to get information
    with open(model_file.name, 'rb') as f:
      model_data = pickle.load(f)
    model_name = model_data.get('name', 'Unknown')
    selected_features = model_data.get('selected_features', [])
    # Generate comprehensive prediction code
    code = f"""# Prediction Code for {model_name}
# Generated by ML Regression Dashboard
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
def load_model(model_path):
  \"\"\"Load the trained model\"\"\"
  with open(model_path, 'rb') as f:
    model_data = pickle.load(f)
  return model_data
def preprocess_data(df, selected_features):
  \"\"\"Preprocess the input data to match training format\"\"\"
  # Make a copy to avoid modifying original
  df_processed = df.copy()
  # Handle missing values
  for col in selected_features:
    if col in df_processed.columns:
      if df_processed[col].isna().sum() > 0:
        if df_processed[col].dtype in ['object', 'category']:
          # Fill categorical with 'Unknown'
          df_processed[col] = df_processed[col].fillna("Unknown")
        else:
          # Fill numerical with median
          df_processed[col] = df_processed[col].fillna(df_processed[col].median())
  # Handle categorical features (simple label encoding)
  for col in selected_features:
    if col in df_processed.columns:
      if df_processed[col].dtype in ['object', 'category']:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
  # Select only required features
  try:
    return df_processed[selected_features]
  except KeyError as e:
    missing_features = [f for f in selected_features if f not in df_processed.columns]
    raise ValueError(f"Missing required features: {{missing_features}}")
def make_predictions(model_path, test_data_path, output_path=None):
  \"\"\"
  Make predictions using the trained model
  Args:
    model_path (str): Path to the saved model (.pkl file)
    test_data_path (str): Path to test data (CSV or Excel file)
    output_path (str, optional): Path to save predictions
  Returns:
    pandas.DataFrame: Test data with predictions added
  \"\"\"
  # Load the model
  print("Loading model...")
  model_data = load_model(model_path)
  model = model_data['model']
  selected_features = model_data['selected_features']
  model_name = model_data['name']
  print(f"Model: {{model_name}}")
  print(f"Required features: {{len(selected_features)}}")
  print("Features:", selected_features[:5], "..." if len(selected_features) > 5 else "")
  # Load test data
  print("Loading test data...")
  if test_data_path.endswith('.csv'):
    test_data = pd.read_csv(test_data_path)
  elif test_data_path.endswith(('.xls', '.xlsx')):
    test_data = pd.read_excel(test_data_path)
  else:
    raise ValueError("Unsupported file format. Use CSV or Excel files.")
  print(f"Test data shape: {{test_data.shape}}")
  # Preprocess the data
  print("Preprocessing data...")
  X_test = preprocess_data(test_data, selected_features)
  print(f"Processed data shape: {{X_test.shape}}")
  # Make predictions
  print("Making predictions...")
  predictions = model.predict(X_test)
  # Add predictions to original data
  result_df = test_data.copy()
  result_df['Prediction'] = predictions
  print(f"Predictions completed!")
  print(f"Prediction range: {{predictions.min():.4f}} to {{predictions.max():.4f}}")
  print(f"Average prediction: {{predictions.mean():.4f}}")
  # Save if output path provided
  if output_path:
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to: {{output_path}}")
  return result_df
# Example usage:
if __name__ == "__main__":
  # Update these paths
  MODEL_PATH = "path/to/your/model.pkl"
  TEST_DATA_PATH = "path/to/your/test_data.csv"
  OUTPUT_PATH = "path/to/save/predictions.csv" # Optional
  try:
    # Make predictions
    results = make_predictions(MODEL_PATH, TEST_DATA_PATH, OUTPUT_PATH)
    # Display first few predictions
    print("\\nFirst 5 predictions:")
    print(results[['Prediction']].head())
  except Exception as e:
    print(f"Error: {{e}}")
# Required features for this model:
# {selected_features}
# Model Information:
# - Name: {model_name}
# - Features: {len(selected_features)}
# - Export Date: {model_data.get('export_date', 'Unknown')}
"""
    return code
  except Exception as e:
    return f"""# Error loading model: {str(e)}
# Please upload a valid model file (.pkl) to see the prediction code"""
# === GRADIO APP ===
def create_ml_dashboard():
  """Create Gradio app with integrated functionality and manual parameter tuning"""
  # Define default parameter values
  default_params = {
    "lgbm": {
      "n_estimators": 2000,
      "num_leaves": 64,
      "max_depth": 8,
      "learning_rate": 0.03,
      "colsample_bytree": 0.9,
      "subsample": 0.8,
      "subsample_freq": 5,
      "min_child_samples": 20
    },
    "xgb": {
      "n_estimators": 1000,
      "max_depth": 8,
      "learning_rate": 0.03,
      "subsample": 0.8,
      "colsample_bytree": 0.9
    },
    "rf": {
      "n_estimators": 500,
      "max_depth": 12,
      "min_samples_leaf": 5
    },
    "catboost": {
      "iterations": 1000,
      "depth": 6,
      "learning_rate": 0.03,
      "l2_leaf_reg": 3,
      "rsm": 0.8,
      "bagging_temperature": 1
    },
    "gam": {
      "n_splines": 10,
      "spline_order": 3,
      "lam": 0.1,
      "n_estimators": 100
    },
    "nn": {
      "hidden_layers": [64, 32],
      "activation": "relu",
      "learning_rate": 0.001,
      "epochs": 100,
      "batch_size": 32,
      "dropout_rate": 0.2
    },
    "hpo_dnn": {
      "max_layers": 5,
      "max_units": 256,
      "epochs": 150,
      "learning_rate": 0.001
    },
    "ensemble": {
      "use_meta_learner": True,
      "include_baselines": False
    }
  }
  # Create the Gradio interface
  with gr.Blocks(title="ML Regression Dashboard", theme='Taithrah/Minimal') as demo:
    gr.Markdown("""
    # 🚀 Advanced ML Regression Dashboard
    Upload your dataset, select models, tune hyperparameters manually, get comprehensive regression analysis with visualizations, and make predictions with your trained models.
    """)
    # State to hold the uploaded dataframe
    state = gr.State({
      "df": None,
      "X": None,
      "y": None,
      "stats": None,
      "model_params": default_params
    })
    with gr.Tab("1️⃣ Upload & Configure"):
      with gr.Row():
        with gr.Column():
          file_input = gr.File(
            label="Upload Dataset (CSV or Excel)",
            file_types=[".csv", ".xlsx", ".xls"]
          )
          upload_btn = gr.Button("Upload & Preview", variant="primary")
          upload_status = gr.Markdown("No file uploaded yet")
        preview_df = gr.Dataframe(label="Data Preview")
      with gr.Row():
        with gr.Column():
          target_dropdown = gr.Dropdown(
            label="Select Target Variable",
            choices=[],
            interactive=True
          )
          ignore_columns = gr.CheckboxGroup(
            label="Select Columns to Ignore",
            choices=[],
            interactive=True
          )
        column_info = gr.Dataframe(label="Column Information")
    with gr.Tab("2️⃣ Model Selection & Configuration"):
      gr.Markdown("### 🔧 Model Selection & Manual Parameter Tuning")
      with gr.Row():
        with gr.Column(scale=1):
          model_selection = gr.CheckboxGroup(
            label="Select Models to Train",
            choices=[
              "LightGBM", "XGBoost", "RandomForest", "CatBoost",
              "GradientBoostedGAM", "BestFitNN", "HPO_DNN",
              "WeightedEnsemble", "PredictiveEnsemble", "HighPerformanceEnsemble",
              "MeanBaseline", "MedianBaseline"
            ],
            value=["LightGBM", "XGBoost"], # Default selection
            interactive=True
          )
          feature_selection = gr.Slider(
            minimum=0, maximum=90, value=40, step=5,
            label="Feature Selection Intensity (%)",
            info="Higher = more aggressive feature reduction"
          )
          use_boruta = gr.Checkbox(
            label="Include Boruta Feature Selection",
            value=False,
            info="Boruta provides high-quality feature selection but takes much longer"
          )
        with gr.Column(scale=2):
          gr.Markdown("### 🎛️ Manual Parameter Tuning")
          gr.Markdown("*Parameters will appear based on your model selection above*")
          # LightGBM Parameters
          with gr.Accordion("🟢 LightGBM Parameters", open=False, visible=False) as lgbm_accordion:
            lgbm_n_estimators = gr.Slider(100, 5000, value=2000, step=100, label="n_estimators")
            lgbm_num_leaves = gr.Slider(10, 300, value=64, step=5, label="num_leaves")
            lgbm_max_depth = gr.Slider(3, 15, value=8, step=1, label="max_depth")
            lgbm_learning_rate = gr.Slider(0.01, 0.3, value=0.03, step=0.01, label="learning_rate")
            lgbm_colsample_bytree = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="colsample_bytree")
            lgbm_subsample = gr.Slider(0.5, 1.0, value=0.8, step=0.05, label="subsample")
            lgbm_min_child_samples = gr.Slider(5, 100, value=20, step=5, label="min_child_samples")
          # XGBoost Parameters
          with gr.Accordion("🔵 XGBoost Parameters", open=False, visible=False) as xgb_accordion:
            xgb_n_estimators = gr.Slider(100, 5000, value=1000, step=100, label="n_estimators")
            xgb_max_depth = gr.Slider(3, 15, value=8, step=1, label="max_depth")
            xgb_learning_rate = gr.Slider(0.01, 0.3, value=0.03, step=0.01, label="learning_rate")
            xgb_subsample = gr.Slider(0.5, 1.0, value=0.8, step=0.05, label="subsample")
            xgb_colsample_bytree = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="colsample_bytree")
          # Random Forest Parameters
          with gr.Accordion("🟠 Random Forest Parameters", open=False, visible=False) as rf_accordion:
            rf_n_estimators = gr.Slider(100, 2000, value=500, step=50, label="n_estimators")
            rf_max_depth = gr.Slider(5, 30, value=12, step=1, label="max_depth")
            rf_min_samples_leaf = gr.Slider(1, 20, value=5, step=1, label="min_samples_leaf")
          # CatBoost Parameters
          with gr.Accordion("🟡 CatBoost Parameters", open=False, visible=False) as cat_accordion:
            cat_iterations = gr.Slider(100, 5000, value=1000, step=100, label="iterations")
            cat_depth = gr.Slider(3, 10, value=6, step=1, label="depth")
            cat_learning_rate = gr.Slider(0.01, 0.3, value=0.03, step=0.01, label="learning_rate")
            cat_l2_leaf_reg = gr.Slider(1, 10, value=3, step=1, label="l2_leaf_reg")
            cat_rsm = gr.Slider(0.5, 1.0, value=0.8, step=0.05, label="rsm")
          # Neural Network Parameters
          with gr.Accordion("🔴 Neural Network Parameters", open=False, visible=False) as nn_accordion:
            nn_epochs = gr.Slider(50, 500, value=100, step=10, label="epochs")
            nn_learning_rate = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="learning_rate")
            nn_batch_size = gr.Slider(16, 128, value=32, step=16, label="batch_size")
            nn_dropout_rate = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="dropout_rate")
      # Hyperparameter tuning options
      with gr.Row():
        with gr.Column():
          gr.Markdown("### 🎯 Hyperparameter Tuning")
          use_hp_tuning = gr.Checkbox(
            label="Enable Hyperparameter Tuning",
            value=False,
            info="Automatically find optimal parameters (increases training time 10-15x)"
          )
          hp_tuning_type = gr.Radio(
            choices=["grid", "random"],
            value="random",
            label="Tuning Method",
            info="Random Search is much faster and usually sufficient"
          )
      with gr.Row():
        reset_params_btn = gr.Button("🔄 Reset to Default Parameters", variant="secondary")
        apply_params_btn = gr.Button("✅ Apply Parameter Changes", variant="primary")
      params_status = gr.Markdown("Select models above to see their parameter options")
    with gr.Tab("3️⃣ Train & Analyze"):
      with gr.Row():
        analyze_btn = gr.Button("🔥 Start Training & Analysis", variant="primary", size="lg")
      # Enhanced console output for detailed logs
      # Enhanced console output for detailed logs
      analysis_logs = gr.Textbox(
        label="Real-time Training Log",
        placeholder="Training logs will appear here in real-time...",
        lines=15,
        max_lines=20,
        autoscroll=True,
        show_copy_button=True,
        container=True,
        interactive=False,
        elem_classes=["log-container"]
      )
      analysis_status = gr.Markdown("Configure settings and start analysis when ready")
      with gr.TabItem("Model Results"):
        # Model summary at the top
        model_summary = gr.Markdown()
        # Results table taking full width with proper interactivity
        results_table = gr.Dataframe(
          label="Model Performance Comparison",
          interactive=True,
          wrap=True,
          max_height=600,
          column_widths=["25%", "15%", "15%", "15%", "15%", "15%"],
          datatype=["str", "number", "number", "number", "number", "number"]
        )
        with gr.TabItem("Feature Selection"):
          feature_plot = gr.Plot()
        with gr.TabItem("Model Performance"):
          performance_plot = gr.Plot()
        with gr.TabItem("Feature Importance"):
          importance_plot = gr.Plot()
    with gr.Tab("4️⃣ Export & Import"):
      gr.Markdown("""
      ### 📥📤 Export & Import Analysis Configurations
      **Export** your entire analysis configuration to share with others or save for later use.
      **Import** configurations from previous analyses to quickly reproduce results.
      """)
      with gr.Row():
        with gr.Column():
          gr.Markdown("#### 📤 Export Configuration")
          with gr.Row():
            export_best_model_btn = gr.Button("Export Best Model", variant="primary")
            export_schema_btn = gr.Button("Export Full Analysis Schema", variant="primary")
          export_status = gr.Markdown("Select what you want to export")
          with gr.Row():
            best_model_download = gr.File(label="Download Best Model (.pkl)")
            schema_download = gr.File(label="Download Analysis Schema (.json)")
        with gr.Column():
          gr.Markdown("#### 📥 Import Configuration")
          schema_input = gr.File(
            label="Upload Analysis Schema (.json)",
            file_types=[".json"]
          )
          import_schema_btn = gr.Button("Import & Apply Schema", variant="secondary")
          import_status = gr.Markdown("Upload a schema file to import configuration")
      with gr.Accordion("🔍 Individual Model Export & Comparison", open=False):
        with gr.Row():
          model_selector = gr.Dropdown(
            label="Select Model to View Details",
            choices=[],
            interactive=True
          )
          export_selected_btn = gr.Button("Export Selected Model", variant="secondary")
        model_details = gr.Markdown("Select a model to view details")
        selected_model_performance = gr.Plot(label="Selected Model Performance")
        selected_model_download = gr.File(label="Download Selected Model (.pkl)")
      with gr.Accordion("📋 Analysis Summary", open=False):
        analysis_summary = gr.Markdown("Run an analysis first to see the summary")
    with gr.Tab("5️⃣ Make Predictions"):
      gr.Markdown("""
      ### 🔮 Make Predictions with Your Trained Model
      Upload your trained model (.pkl file) and new data to generate predictions.
      """)
      with gr.Row():
        with gr.Column():
          prediction_model_input = gr.File(
            label="Upload Saved Model (.pkl)",
            file_types=[".pkl"]
          )
          prediction_data_input = gr.File(
            label="Upload Test Data (CSV or Excel)",
            file_types=[".csv", ".xlsx", ".xls"]
          )
          predict_btn = gr.Button("Generate Predictions", variant="primary", size="lg")
        prediction_status = gr.Markdown("Upload a model and test data to generate predictions")
      # Add console output for prediction logs
      prediction_logs = gr.Textbox(
        label="Prediction Log",
        placeholder="Prediction logs will appear here...",
        lines=5,
        max_lines=10,
        autoscroll=True
      )
      with gr.Row():
        prediction_preview = gr.Dataframe(label="Test Data Preview (First 5 Rows)")
      with gr.Row():
        prediction_results = gr.Dataframe(label="Prediction Results")
        prediction_download = gr.File(label="Download Predictions")
      # NEW: Prediction Code Display
      with gr.Accordion("📋 Copy Prediction Code", open=False):
        gr.Markdown("""
        ### 🐍 Python Code for Making Predictions
        Copy the code below to use your trained model anywhere outside this dashboard.
        Make sure you have the required libraries installed: `pip install pandas numpy scikit-learn`
        """)
        prediction_code = gr.Code(
          label="Prediction Code (Copy & Use)",
          language="python",
          value="# Upload a model first to see the prediction code",
          lines=20
        )
    # Functions for data upload and processing
    def load_file(file):
      if not file:
        return (
          gr.Markdown("No file uploaded yet"),
          {"df": None},
          gr.Dataframe(),
          gr.Dataframe(),
          gr.Dropdown(choices=[]),
          gr.CheckboxGroup(choices=[])
        )
      try:
        # Load the dataset
        if file.name.endswith('.csv'):
          df = pd.read_csv(file.name)
        elif file.name.endswith(('.xls', '.xlsx')):
          df = pd.read_excel(file.name)
        else:
          return (
            gr.Markdown("⚠️ Unsupported file format. Please upload a CSV or Excel file."),
            {"df": None},
            gr.Dataframe(),
            gr.Dataframe(),
            gr.Dropdown(choices=[]),
            gr.CheckboxGroup(choices=[])
          )
        # Basic validation
        if df.empty:
          return (
            gr.Markdown("⚠️ Dataset is empty."),
            {"df": None},
            gr.Dataframe(),
            gr.Dataframe(),
            gr.Dropdown(choices=[]),
            gr.CheckboxGroup(choices=[])
          )
        # Update state
        state_data = {"df": df, "model_params": default_params}
        # Analyze columns
        col_info = []
        for col in df.columns:
          dtype = df[col].dtype
          missing = df[col].isna().sum()
          unique_count = df[col].nunique()
          col_info.append({
            "Column": col,
            "Type": str(dtype),
            "Missing": missing,
            "Unique Values": unique_count,
            "% Missing": f"{missing/len(df)*100:.1f}%"
          })
        col_info_df = pd.DataFrame(col_info)
        # Suggest target column (first numeric)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_choices = df.columns.tolist()
        suggested_target = numeric_cols[0] if numeric_cols else target_choices[0]
        # Suggest columns to ignore
        ignore_suggestions = []
        for col in df.columns:
          col_lower = col.lower()
          if any(keyword in col_lower for keyword in ['id', 'code', 'name', 'date', 'time']):
            ignore_suggestions.append(col)
        # Also suggest columns with almost unique values (likely IDs)
        for col in df.columns:
          if df[col].nunique() > 0.9 * len(df):
            if col not in ignore_suggestions:
              ignore_suggestions.append(col)
        # Success message
        upload_message = f"""
        ✅ Dataset loaded successfully!
        **Rows:** {df.shape[0]:,}
        **Columns:** {df.shape[1]}
        **Memory Usage:** {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB
        **Missing Values:** {df.isna().sum().sum():,} ({df.isna().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%)
        Preview the first few rows below and configure your analysis.
        """
        return (
          gr.Markdown(upload_message),
          state_data,
          df.head(10),
          col_info_df,
          gr.Dropdown(choices=target_choices, value=suggested_target),
          gr.CheckboxGroup(choices=target_choices, value=ignore_suggestions)
        )
      except Exception as e:
        return (
          gr.Markdown(f"⚠️ Error loading file: {str(e)}"),
          {"df": None},
          gr.Dataframe(),
          gr.Dataframe(),
          gr.Dropdown(choices=[]),
          gr.CheckboxGroup(choices=[])
        )
    # Functions for parameter management
    def update_parameter_visibility(selected_models):
      """Update parameter accordion visibility based on selected models"""
      updates = []
      # Define all model accordions
      all_models = ["LightGBM", "XGBoost", "RandomForest", "CatBoost", "BestFitNN"]
      for model in all_models:
        visible = model in selected_models
        updates.append(gr.Accordion(visible=visible))
      # Update status message
      if selected_models:
        param_models = [m for m in selected_models if m in all_models]
        if param_models:
          status = f"✅ Parameter tuning available for: {', '.join(param_models)}"
        else:
          status = "ℹ️ Selected models don't have manual parameter tuning options"
      else:
        status = "Select models above to see their parameter options"
      updates.append(gr.Markdown(status))
      return updates
    def apply_parameter_changes(lgbm_n_est, lgbm_leaves, lgbm_depth, lgbm_lr, lgbm_col, lgbm_sub, lgbm_child,
                 xgb_n_est, xgb_depth, xgb_lr, xgb_sub, xgb_col,
                 rf_n_est, rf_depth, rf_leaf,
                 cat_iter, cat_depth, cat_lr, cat_l2, cat_rsm,
                 nn_epochs, nn_lr, nn_batch, nn_dropout, state_data):
      """Apply parameter changes to state"""
      new_params = {
        "lgbm": {
          "n_estimators": int(lgbm_n_est),
          "num_leaves": int(lgbm_leaves),
          "max_depth": int(lgbm_depth),
          "learning_rate": lgbm_lr,
          "colsample_bytree": lgbm_col,
          "subsample": lgbm_sub,
          "subsample_freq": 5,
          "min_child_samples": int(lgbm_child)
        },
        "xgb": {
          "n_estimators": int(xgb_n_est),
          "max_depth": int(xgb_depth),
          "learning_rate": xgb_lr,
          "subsample": xgb_sub,
          "colsample_bytree": xgb_col
        },
        "rf": {
          "n_estimators": int(rf_n_est),
          "max_depth": int(rf_depth),
          "min_samples_leaf": int(rf_leaf)
        },
        "catboost": {
          "iterations": int(cat_iter),
          "depth": int(cat_depth),
          "learning_rate": cat_lr,
          "l2_leaf_reg": int(cat_l2),
          "rsm": cat_rsm,
          "bagging_temperature": 1
        },
        "nn": {
          "hidden_layers": [64, 32],
          "activation": "relu",
          "learning_rate": nn_lr,
          "epochs": int(nn_epochs),
          "batch_size": int(nn_batch),
          "dropout_rate": nn_dropout
        },
        "gam": state_data.get("model_params", default_params)["gam"],
        "hpo_dnn": state_data.get("model_params", default_params)["hpo_dnn"],
        "ensemble": state_data.get("model_params", default_params)["ensemble"]
      }
      # Update state
      updated_state = state_data.copy()
      updated_state["model_params"] = new_params
      return updated_state, gr.Markdown("✅ Parameters updated successfully! Ready for training.")
    def reset_parameters():
      """Reset all parameters to default values"""
      params = default_params
      return (
        # LightGBM
        params["lgbm"]["n_estimators"], params["lgbm"]["num_leaves"], params["lgbm"]["max_depth"],
        params["lgbm"]["learning_rate"], params["lgbm"]["colsample_bytree"], params["lgbm"]["subsample"], params["lgbm"]["min_child_samples"],
        # XGBoost
        params["xgb"]["n_estimators"], params["xgb"]["max_depth"], params["xgb"]["learning_rate"],
        params["xgb"]["subsample"], params["xgb"]["colsample_bytree"],
        # Random Forest
        params["rf"]["n_estimators"], params["rf"]["max_depth"], params["rf"]["min_samples_leaf"],
        # CatBoost
        params["catboost"]["iterations"], params["catboost"]["depth"], params["catboost"]["learning_rate"],
        params["catboost"]["l2_leaf_reg"], params["catboost"]["rsm"],
        # Neural Network
        params["nn"]["epochs"], params["nn"]["learning_rate"], params["nn"]["batch_size"], params["nn"]["dropout_rate"],
        # Status
        gr.Markdown("🔄 Parameters reset to default values")
      )
    def run_analysis(target_col, ignore_cols, feature_selection_intensity, selected_models,
            use_hp_tuning, hp_tuning_type, use_boruta_flag, state_data, progress=gr.Progress()):
      """Run the complete ML analysis pipeline with model selection and hyperparameter tuning"""
      global training_results
      if not state_data.get("df") is not None:
        return (
          "", # clear logs
          gr.Markdown("⚠️ Please upload a dataset first."),
          None, None, None, None, gr.Dropdown(choices=[])
        )
      if not target_col:
        return (
          "", # clear logs
          gr.Markdown("⚠️ Please select a target variable."),
          None, None, None, None, gr.Dropdown(choices=[])
        )
      if not selected_models:
        return (
          "", # clear logs
          gr.Markdown("⚠️ Please select at least one model to train."),
          None, None, None, None, gr.Dropdown(choices=[])
        )
      try:
        df = state_data["df"]
        model_params = state_data.get("model_params", default_params)
        # Initialize log content
        log_content = ""
        # Function to update logs with real-time display
        # Function to update logs with real-time display
        def update_log(message):
          nonlocal log_content
          timestamp = datetime.now().strftime("%H:%M:%S")
          new_line = f"[{timestamp}] {message}"
          if log_content:
            log_content = log_content + "\n" + new_line
          else:
            log_content = new_line
          return log_content
        # Update status
        try:
          if progress is not None:
            progress(0.05, desc="Initializing analysis...")
        except Exception as e:
          pass # Continue silently if progress fails
        yield (
          update_log("🚀 Analysis started - preparing environment"),
          gr.Markdown("🔄 Analysis in progress..."),
          None, None, None, None, gr.Dropdown(choices=[])
        )
        # Log configuration
        update_log(f"🎯 Selected models: {', '.join(selected_models)}")
        update_log(f"🎛️ Feature selection intensity: {feature_selection_intensity}%")
        if use_boruta_flag:
          update_log("🧬 Boruta feature selection enabled (this will take longer)")
        else:
          update_log("⚡ Boruta disabled for faster execution")
        if use_hp_tuning:
          update_log(f"⚙️ Hyperparameter tuning: {hp_tuning_type} search enabled")
          update_log("⚠️ Training time will be significantly increased due to hyperparameter tuning")
        # Preprocess data with logging
        try:
          if progress is not None:
            progress(0.1, desc="Preprocessing data...")
        except Exception as e:
          pass # Continue silently if progress fails
        yield (
          update_log("🔍 Starting data preprocessing..."),
          gr.Markdown("🔄 Analysis in progress...\\n\\n**Step 1:** Preprocessing data..."),
          None, None, None, None, gr.Dropdown(choices=[])
        )
        X, y, stats, error = preprocess_data(df, target_col, ignore_cols,
                          log_callback=update_log)
        if error:
          yield (
            update_log(f"❌ Error: {error}"),
            gr.Markdown(f"⚠️ Error: {error}"),
            None, None, None, None, gr.Dropdown(choices=[])
          )
          return
        # Run analysis with progress and logging
        try:
          if progress is not None:
            progress(0.2, desc="Starting feature selection and model training...")
        except Exception as e:
          pass # Continue silently if progress fails
        yield (
          update_log("⚡ Starting parallel feature selection and model training..."),
          gr.Markdown("🔄 Analysis in progress...\\n\\n**Step 2:** Running feature selection and training models in parallel..."),
          None, None, None, None, gr.Dropdown(choices=[])
        )
        # Pass all the new parameters to the training function
        results = train_and_evaluate_models(
          X, y,
          feature_selection_intensity,
          model_params,
          selected_models, # Pass selected models
          use_hp_tuning, # Pass hyperparameter tuning flag
          hp_tuning_type, # Pass tuning type
          use_boruta_flag, # NEW: Pass Boruta flag
          progress_callback=progress,
          log_callback=update_log
        )
        # Store results in global variable
        training_results = {
          'results_df': results['results_df'],
          'best_model_name': results['best_model_name'],
          'trained_models': results['trained_models'],
          'selected_features': results['selected_features'],
          'y_test': results['y_test'],
          'y_pred_dict': results['y_pred_dict']
        }
        # Create visualizations
        try:
          if progress is not None:
            progress(0.9, desc="Creating visualizations...")
        except Exception as e:
          pass # Continue silently if progress fails
        yield (
          update_log("📊 Creating visualizations..."),
          gr.Markdown("🔄 Analysis in progress...\\n\\n**Step 3:** Creating visualizations..."),
          None, None, None, None, gr.Dropdown(choices=[])
        )
        fs_fig = visualize_feature_selection(
          results['X_for_importance'],
          results['votes_df'],
          results['selected_features']
        )
        perf_fig = visualize_model_performance(
          results['results_df'],
          results['y_test'],
          results['y_pred_dict']
        )
        fi_fig, importance_df = visualize_feature_importance(
          results['X_for_importance'],
          results['best_model'],
          results['best_model_name']
        )
        # Log top features by importance
        if importance_df is not None:
          top_5 = importance_df.head(5)['Feature'].tolist()
          update_log(f"🏆 Top 5 important features: {', '.join(top_5)}")
        # Create summary
        wape_value = results['results_df'].loc[results['best_model_name'], 'WAPE']
        r2_value = results['results_df'].loc[results['best_model_name'], 'R2']
        tuning_info = ""
        if use_hp_tuning and any("HP-" in model for model in results['results_df'].index):
          tuning_info = f"\\n**Hyperparameter Tuning:** {hp_tuning_type} search completed for applicable models"
        boruta_info = ""
        if use_boruta_flag:
          boruta_info = "\\n**Feature Selection:** Boruta included for high-quality feature selection"
        summary = f"""
        ### 🎉 Analysis Complete!
        **Best Model:** {results['best_model_name']}
        - **WAPE:** {wape_value:.4f}
        - **R² Score:** {r2_value:.4f}
        **Models Trained:** {', '.join(selected_models)}{tuning_info}{boruta_info}
        **Feature Selection:**
        - Selected {len(results['selected_features'])} out of {X.shape[1]} features
        - Top feature: {results['votes_df'].sort_values('Score', ascending=False).index[0]}
        **Dataset Info:**
        - Rows: {stats['rows']}
        - Target range: {stats['target_min']:.2f} to {stats['target_max']:.2f}
        **Ready for Export & Predictions:**
        - Go to the 'Export & Import' tab to download models and configurations
        - Use the 'Make Predictions' tab to predict on new data
        """
        try:
          if progress is not None:
            progress(1.0, desc="Analysis complete!")
        except Exception as e:
          pass # Continue silently if progress update fails
        update_log("✅ Analysis complete!")
        # Get model names for dropdown
        model_names = list(results['results_df'].index)
        # Final result
        yield (
          log_content,
          summary,
          results['results_df'].reset_index(),
          fs_fig,
          perf_fig,
          fi_fig,
          gr.Dropdown(choices=model_names, value=model_names[0] if model_names else None)
        )
      except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_message = f"❌ Error during analysis: {str(e)}\\n\\n{error_details}"
        yield (
          update_log(error_message),
          gr.Markdown(f"⚠️ Error during analysis: {str(e)}"),
          None, None, None, None, gr.Dropdown(choices=[])
        )
    # Function to handle prediction (unchanged)
    def load_test_data(file):
      """Load test data for prediction"""
      if not file:
        return None, "No file uploaded"
      try:
        if file.name.endswith('.csv'):
          df = pd.read_csv(file.name)
        elif file.name.endswith(('.xls', '.xlsx')):
          df = pd.read_excel(file.name)
        else:
          return None, "Unsupported file format"
        return df, None
      except Exception as e:
        return None, f"Error loading test data: {str(e)}"
    def run_predictions(model_file, data_file):
      """Generate predictions using the uploaded model and test data"""
      # Initialize log content
      log_content = ""
      # Function to update logs
      def update_log(message):
        nonlocal log_content
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_content += f"[{timestamp}] {message}\\n"
        return log_content
      if not model_file or not data_file:
        return (
          update_log("⚠️ Please upload both a model file and test data."),
          gr.Markdown("⚠️ Please upload both a model file and test data."),
          None,
          None,
          None
        )
      try:
        # Load model
        update_log(f"📂 Loading model from {model_file.name}")
        model_data, model_error = load_model(model_file.name)
        if model_error:
          return (
            update_log(f"❌ {model_error}"),
            gr.Markdown(f"⚠️ {model_error}"),
            None,
            None,
            None
          )
        update_log(f"✅ Model loaded successfully: {model_data['name']}")
        # Load test data
        update_log(f"📂 Loading test data from {data_file.name}")
        test_data, data_error = load_test_data(data_file)
        if data_error:
          return (
            update_log(f"❌ {data_error}"),
            gr.Markdown(f"⚠️ {data_error}"),
            test_data.head(5) if test_data is not None else None,
            None,
            None
          )
        update_log(f"✅ Test data loaded successfully: {test_data.shape[0]} rows, {test_data.shape[1]} columns")
        # Make predictions
        update_log("🔮 Generating predictions...")
        results, pred_error = predict_with_model(model_data, test_data, log_callback=update_log)
        if pred_error:
          return (
            update_log(f"❌ {pred_error}"),
            gr.Markdown(f"⚠️ {pred_error}"),
            test_data.head(5) if test_data is not None else None,
            None,
            None
          )
        # Save predictions to file
        output_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        results.to_csv(pred_path, index=False)
        update_log(f"💾 Saved predictions to {pred_path}")
        # Compute basic prediction statistics
        pred_min = results['Prediction'].min()
        pred_max = results['Prediction'].max()
        pred_mean = results['Prediction'].mean()
        pred_std = results['Prediction'].std()
        update_log(f"📊 Prediction statistics: Min={pred_min:.4f}, Max={pred_max:.4f}, Mean={pred_mean:.4f}, Std={pred_std:.4f}")
        update_log("✅ Prediction process complete!")
        success_message = f"""
        ### ✅ Predictions Generated Successfully!
        **Model:** {model_data['name']}
        **Features used:** {len(model_data['selected_features'])}
        **Predictions:** {len(results)} rows
        **Prediction Statistics:**
        - Min: {pred_min:.4f}
        - Max: {pred_max:.4f}
        - Mean: {pred_mean:.4f}
        - Std Dev: {pred_std:.4f}
        Your predictions are ready to download!
        """
        return (
          log_content,
          gr.Markdown(success_message),
          test_data.head(5),
          results,
          pred_path
        )
      except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_message = f"❌ Error during prediction: {str(e)}\\n\\n{error_details}"
        return (
          update_log(error_message),
          gr.Markdown(f"⚠️ Error during prediction: {str(e)}"),
          None,
          None,
          None
        )
    # Export/Import Functions (unchanged)
    def handle_export_best_model():
      """Export the best model"""
      global training_results
      if not training_results:
        return None, gr.Markdown("⚠️ No analysis results available. Run an analysis first.")
      best_model_name = training_results['best_model_name']
      model_path, error = export_selected_model(best_model_name)
      if error:
        return None, gr.Markdown(f"⚠️ {error}")
      return model_path, gr.Markdown(f"✅ Best model '{best_model_name}' exported successfully!")
    def handle_export_schema(feature_selection_intensity, target_col, ignore_cols, state_data):
      """Export full analysis schema"""
      model_params = state_data.get("model_params", default_params)
      schema_path, error = export_analysis_schema(feature_selection_intensity, target_col, ignore_cols, model_params)
      if error:
        return None, gr.Markdown(f"⚠️ {error}")
      return schema_path, gr.Markdown("✅ Analysis schema exported successfully!")
    def handle_import_schema(schema_file):
      """Import analysis schema and update UI"""
      config, error = import_analysis_schema(schema_file)
      if error:
        return (
          gr.Markdown(f"⚠️ {error}"),
          gr.Dropdown(),
          gr.CheckboxGroup(),
          gr.Slider()
        )
      message = f"""
      ✅ Schema imported successfully!
      **Configuration Loaded:**
      - Target column: {config['target_col']}
      - Feature selection intensity: {config['feature_selection_intensity']}%
      - Ignored columns: {', '.join(config['ignore_cols']) if config['ignore_cols'] else 'None'}
      - Model parameters: Updated to imported values
      **Next Steps:**
      1. Go to 'Upload & Configure' tab to upload your dataset
      2. Verify the target column and ignored columns are set correctly
      3. Go to 'Train & Analyze' tab to run the analysis with imported settings
      """
      return (
        gr.Markdown(message),
        gr.Dropdown(value=config['target_col']),
        gr.CheckboxGroup(value=config['ignore_cols']),
        gr.Slider(value=config['feature_selection_intensity'])
      )
    def handle_model_selection(model_name):
      """Handle individual model selection"""
      details, plot = get_model_details(model_name)
      return details, plot
    def handle_export_selected_model(model_name):
      """Export specific selected model"""
      model_path, error = export_selected_model(model_name)
      if error:
        return None, gr.Markdown(f"⚠️ {error}")
      return model_path, gr.Markdown(f"✅ Model '{model_name}' exported successfully!")
    def generate_analysis_summary():
      """Generate analysis summary"""
      global training_results
      if not training_results:
        return "No analysis results available. Run an analysis first."
      results_df = training_results['results_df']
      best_model = training_results['best_model_name']
      selected_features = training_results['selected_features']
      # Create summary
      summary = f"""
      ### 📊 Analysis Summary
      **Best Performing Model:** {best_model}
      - WAPE: {results_df.loc[best_model, 'WAPE']:.4f}
      - R²: {results_df.loc[best_model, 'R2']:.4f}
      - MAE: {results_df.loc[best_model, 'MAE']:.4f}
      - RMSE: {results_df.loc[best_model, 'RMSE']:.4f}
      **Feature Selection:**
      - Selected Features: {len(selected_features)}
      - Top 5 Features: {', '.join(selected_features[:5])}
      **All Models Tested:** {len(results_df)}
      **Model Rankings (by WAPE):**
      """
      # Add top 5 models
      top_5_models = results_df.sort_values('WAPE').head(5)
      for i, (model_name, row) in enumerate(top_5_models.iterrows(), 1):
        summary += f"\\n{i}. {model_name} - WAPE: {row['WAPE']:.4f}"
      return summary
    # Connect events
    upload_btn.click(
      load_file,
      inputs=[file_input],
      outputs=[
        upload_status,
        state,
        preview_df,
        column_info,
        target_dropdown,
        ignore_columns
      ]
    )
    # Parameter visibility updates
    model_selection.change(
      fn=update_parameter_visibility,
      inputs=[model_selection],
      outputs=[lgbm_accordion, xgb_accordion, rf_accordion, cat_accordion, nn_accordion, params_status]
    )
    # Parameter application
    apply_params_btn.click(
      fn=apply_parameter_changes,
      inputs=[
        lgbm_n_estimators, lgbm_num_leaves, lgbm_max_depth, lgbm_learning_rate,
        lgbm_colsample_bytree, lgbm_subsample, lgbm_min_child_samples,
        xgb_n_estimators, xgb_max_depth, xgb_learning_rate, xgb_subsample, xgb_colsample_bytree,
        rf_n_estimators, rf_max_depth, rf_min_samples_leaf,
        cat_iterations, cat_depth, cat_learning_rate, cat_l2_leaf_reg, cat_rsm,
        nn_epochs, nn_learning_rate, nn_batch_size, nn_dropout_rate,
        state
      ],
      outputs=[state, params_status]
    )
    # Parameter reset
    reset_params_btn.click(
      fn=reset_parameters,
      inputs=[],
      outputs=[
        lgbm_n_estimators, lgbm_num_leaves, lgbm_max_depth, lgbm_learning_rate,
        lgbm_colsample_bytree, lgbm_subsample, lgbm_min_child_samples,
        xgb_n_estimators, xgb_max_depth, xgb_learning_rate, xgb_subsample, xgb_colsample_bytree,
        rf_n_estimators, rf_max_depth, rf_min_samples_leaf,
        cat_iterations, cat_depth, cat_learning_rate, cat_l2_leaf_reg, cat_rsm,
        nn_epochs, nn_learning_rate, nn_batch_size, nn_dropout_rate,
        params_status
      ]
    )
    # Updated analyze button with Boruta option
    analyze_btn.click(
      fn=run_analysis,
      inputs=[
        target_dropdown,
        ignore_columns,
        feature_selection,
        model_selection, # Selected models
        use_hp_tuning, # HP tuning flag
        hp_tuning_type, # HP tuning type
        use_boruta, # NEW: Boruta flag
        state
      ],
      outputs=[
        analysis_logs,
        analysis_status,
        results_table,
        feature_plot,
        performance_plot,
        importance_plot,
        model_selector
      ]
    )
    # Export/Import events (unchanged)
    export_best_model_btn.click(
      fn=handle_export_best_model,
      inputs=[],
      outputs=[best_model_download, export_status]
    )
    export_schema_btn.click(
      fn=handle_export_schema,
      inputs=[feature_selection, target_dropdown, ignore_columns, state],
      outputs=[schema_download, export_status]
    )
    import_schema_btn.click(
      fn=handle_import_schema,
      inputs=[schema_input],
      outputs=[import_status, target_dropdown, ignore_columns, feature_selection]
    )
    # Individual model selection events (unchanged)
    model_selector.change(
      fn=handle_model_selection,
      inputs=[model_selector],
      outputs=[model_details, selected_model_performance]
    )
    export_selected_btn.click(
      fn=handle_export_selected_model,
      inputs=[model_selector],
      outputs=[selected_model_download, model_details]
    )
    # Prediction events
    predict_btn.click(
      fn=run_predictions,
      inputs=[prediction_model_input, prediction_data_input],
      outputs=[
        prediction_logs,
        prediction_status,
        prediction_preview,
        prediction_results,
        prediction_download
      ]
    )
    # NEW: Prediction code generation
    prediction_model_input.change(
      fn=generate_prediction_code,
      inputs=[prediction_model_input],
      outputs=[prediction_code]
    )
    # Analysis summary (unchanged)
    demo.load(
      fn=generate_analysis_summary,
      inputs=[],
      outputs=[analysis_summary]
    )
  return demo
# Launch the app
if __name__ == "__main__":
  demo = create_ml_dashboard()
  demo.launch()
