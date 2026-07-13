"""
High-Throughput Computation for UHTC Thermal Response Temperature Prediction
===========================================================================
This script performs high-throughput screening of HfB2-ZrC-TaSi2 
formulations using a CatBoost regression model trained on descriptor-based
features. The model predicts thermal response temperature for unseen 
compositions to identify optimal low-temperature formulations.

Requirements:
    - pandas, numpy, catboost, scikit-learn, matplotlib, openpyxl

Usage:
    python high_throughput_computation.py

Input:
    - Training data: Excel file with descriptor features and target T
    - Prediction data: Excel file with formulation descriptors

Output:
    - HfB2_ZrC_TaSi2_T_prediction_results.xlsx
    - prediction_scatter.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. Data Loading
# ============================================================================

def load_training_data(filepath: str) -> tuple:
    """
    Load training data and extract features/target.

    Parameters
    ----------
    filepath : str
        Path to training data Excel file.
        Expected: feature columns first, target column last.

    Returns
    -------
    features : pd.DataFrame
        Feature matrix.
    target : pd.Series
        Target variable (thermal response temperature).
    feature_names : list
        List of feature column names.
    """
    data = pd.read_excel(filepath)
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    feature_names = features.columns.tolist()

    print(f"Training data loaded: {data.shape}")
    print(f"Features: {feature_names}")
    print(f"Target range: [{target.min():.1f}, {target.max():.1f}] °C")

    return features, target, feature_names


def load_prediction_data(filepath: str, feature_names: list) -> pd.DataFrame:
    """
    Load prediction data and align columns with training features.

    Parameters
    ----------
    filepath : str
        Path to prediction data Excel file.
    feature_names : list
        Expected feature names from training data.

    Returns
    -------
    features : pd.DataFrame
        Aligned feature matrix for prediction.
    """
    data = pd.read_excel(filepath)

    # Check column alignment
    if not all(col in data.columns for col in feature_names):
        print("Warning: Prediction data columns do not match training features.")
        print(f"Expected: {feature_names}")
        print(f"Found: {list(data.columns)}")
        # Align by position
        features = data.iloc[:, :len(feature_names)].copy()
        features.columns = feature_names
        print("Aligned by position.")
    else:
        features = data[feature_names].copy()

    print(f"Prediction data loaded: {data.shape}")

    return features


# ============================================================================
# 2. Model Configuration & Training
# ============================================================================

def get_catboost_params() -> dict:
    """
    Return optimized CatBoost hyperparameters for thermal response prediction.
    Tuned for small-to-medium datasets with potential non-linear relationships.
    """
    return {
        'iterations': 300,
        'depth': 7,
        'learning_rate': 0.1,
        'l2_leaf_reg': 7,
        'random_strength': 3,
        'colsample_bylevel': 0.8,
        'bagging_temperature': 1,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }


def train_and_evaluate(X_train: np.ndarray, y_train: pd.Series,
                        params: dict) -> tuple:
    """
    Train CatBoost model and evaluate on training set.

    Returns
    -------
    model : CatBoostRegressor
        Trained model.
    metrics : dict
        Dictionary with R2, MSE, MAE on training data.
    """
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    metrics = {
        'R2': r2_score(y_train, y_pred),
        'MSE': mean_squared_error(y_train, y_pred),
        'MAE': mean_absolute_error(y_train, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_pred))
    }

    print("\n--- Training Set Performance ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, metrics


# ============================================================================
# 3. High-Throughput Prediction
# ============================================================================

def predict_formulations(model: CatBoostRegressor, X_pred: np.ndarray,
                         formulation_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict thermal response temperature for all formulations.

    Parameters
    ----------
    model : CatBoostRegressor
        Trained model.
    X_pred : np.ndarray
        Scaled feature matrix for prediction.
    formulation_data : pd.DataFrame
        Original formulation data (before scaling).

    Returns
    -------
    results : pd.DataFrame
        Formulation data with predicted T appended.
    """
    predictions = model.predict(X_pred)

    results = formulation_data.copy()
    results['Predicted_T_°C'] = predictions

    print(f"\nPredictions: {len(predictions)} formulations")
    print(f"  Min T: {predictions.min():.1f} °C")
    print(f"  Max T: {predictions.max():.1f} °C")
    print(f"  Mean T: {predictions.mean():.1f} °C")
    print(f"  Median T: {np.median(predictions):.1f} °C")

    # Top 10 lowest-T formulations
    top10 = results.nsmallest(10, 'Predicted_T_°C')
    print(f"\nTop 10 lowest-T formulations:")
    print(top10[['Predicted_T_°C'] + [c for c in top10.columns 
                                      if c != 'Predicted_T_°C']].to_string())

    return results


def plot_predictions(results: pd.DataFrame, output_file: str = "prediction_scatter.png"):
    """Plot predicted temperatures across formulations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(range(len(results)), results['Predicted_T_°C'],
               c='#1f77b4', s=20, alpha=0.6, edgecolors='none')
    ax.axhline(y=results['Predicted_T_°C'].mean(), color='red',
               linestyle='--', linewidth=1, label=f"Mean: {results['Predicted_T_°C'].mean():.1f}°C")

    ax.set_xlabel('Formulation Index')
    ax.set_ylabel('Predicted Thermal Response Temperature (°C)')
    ax.set_title('High-Throughput Screening: HfB₂-ZrC-TaSi₂ Formulations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_file}")


# ============================================================================
# 4. Main Execution
# ============================================================================

if __name__ == "__main__":
    # Configuration
    TRAIN_FILE = "descriptor_training_data.xlsx"      # Replace with actual path
    PREDICT_FILE = "HfB2_ZrC_TaSi2_formulations.xlsx"   # Replace with actual path
    OUTPUT_FILE = "HfB2_ZrC_TaSi2_T_prediction_results.xlsx"

    print("=" * 60)
    print("High-Throughput Computation: UHTC Thermal Response Prediction")
    print("=" * 60)

    # Step 1: Load data
    X_train_raw, y_train, feature_names = load_training_data(TRAIN_FILE)
    X_pred_raw = load_prediction_data(PREDICT_FILE, feature_names)

    # Step 2: Min-Max scaling [0, 1]
    print("\nApplying Min-Max scaling [0, 1]...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train_raw)
    X_pred = scaler.transform(X_pred_raw)

    # Step 3: Train model
    params = get_catboost_params()
    model, metrics = train_and_evaluate(X_train, y_train, params)

    # Step 4: High-throughput prediction
    results = predict_formulations(model, X_pred, X_pred_raw)

    # Step 5: Save results
    results.to_excel(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Step 6: Visualization
    plot_predictions(results)

    print("\nDone.")
