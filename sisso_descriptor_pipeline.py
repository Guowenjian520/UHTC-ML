"""
SISSO-like Symbolic Descriptor Construction and Screening
========================================================
This module implements a SISSO (Sure Independence Screening and 
Sparsifying Operator)-like symbolic descriptor construction pipeline
for identifying mathematical expressions that best predict thermal 
response temperature of ultra-high temperature ceramics.

Pipeline:
    1. Feature transformation (reciprocal, power, root transforms)
    2. Feature interaction generation (multiplicative combinations)
    3. Multi-metric screening (Pearson, Spearman, F-test, Mutual Information, RF)
    4. Forward feature selection with XGBoost
    5. Exhaustive symbolic descriptor search

Requirements:
    - pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, openpyxl

Usage:
    python sisso_descriptor_pipeline.py

Output:
    - descriptor_screening_results.xlsx (multi-sheet)
    - XGBoost_RMSE_vs_features.png
    - XGBoost_R2_vs_features.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, permutations
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# PART 1: Feature Engineering & Multi-Metric Screening
# ============================================================================

def build_transformed_features(df: pd.DataFrame, feature_cols: list,
                                target_col: str) -> pd.DataFrame:
    """
    Build comprehensive feature library via physical-motivated transformations.

    For negative-correlation features (CP, TC, Hv, MP, YM, GS, EM, SVP):
        - Reciprocal: 1/x, sqrt(1/x), cbrt(1/x), (1/x)^2, (1/x)^3
    For positive-correlation features (CE, Hr):
        - Power: sqrt(x), cbrt(x), x^2, x^3
    Cross-interactions between transformed negative and positive features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw features and target.
    feature_cols : list
        List of feature column names.
    target_col : str
        Target variable column name.

    Returns
    -------
    final_df : pd.DataFrame
        DataFrame with all original, transformed, and interaction features + target.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Physical-motivated grouping based on correlation direction with T
    negative_features = ['CP', 'TC', 'Hv', 'MP', 'YM', 'GS', 'EM', 'SVP']
    positive_features = ['CE', 'Hr']

    # Validate
    neg_valid = [f for f in negative_features if f in feature_cols]
    pos_valid = [f for f in positive_features if f in feature_cols]

    print(f"Negative-correlation features: {neg_valid}")
    print(f"Positive-correlation features: {pos_valid}")

    # Feature Set 1: Reciprocal transforms for negative-correlation features
    feature_set1 = pd.DataFrame(index=df.index)
    for col in neg_valid:
        inv_col = 1.0 / (X[col] + 1e-10)
        feature_set1[f'inv_{col}'] = inv_col
        feature_set1[f'sqrt_inv_{col}'] = np.sqrt(inv_col)
        feature_set1[f'cbrt_inv_{col}'] = np.cbrt(inv_col)
        feature_set1[f'sq_inv_{col}'] = inv_col ** 2
        feature_set1[f'cube_inv_{col}'] = inv_col ** 3

    # Feature Set 2: Power transforms for positive-correlation features
    feature_set2 = pd.DataFrame(index=df.index)
    for col in pos_valid:
        feature_set2[f'sqrt_{col}'] = np.sqrt(X[col])
        feature_set2[f'cbrt_{col}'] = np.cbrt(X[col])
        feature_set2[f'sq_{col}'] = X[col] ** 2
        feature_set2[f'cube_{col}'] = X[col] ** 3

    # Feature Set 3: Cross-interactions (negative-transformed × positive-transformed)
    feature_set3 = pd.DataFrame(index=df.index)
    for col1 in feature_set1.columns:
        for col2 in feature_set2.columns:
            feature_set3[f'{col1}_x_{col2}'] = feature_set1[col1] * feature_set2[col2]

    # Merge all
    all_features_df = pd.concat([X, feature_set1, feature_set2, feature_set3], axis=1)
    final_df = pd.concat([all_features_df, y], axis=1)

    print(f"\nTotal descriptors constructed: {all_features_df.shape[1]}")
    print(f"  Original: {len(feature_cols)}")
    print(f"  Transformed (negative): {feature_set1.shape[1]}")
    print(f"  Transformed (positive): {feature_set2.shape[1]}")
    print(f"  Interactions: {feature_set3.shape[1]}")

    return final_df


def multi_metric_screening(final_df: pd.DataFrame, target_col: str,
                           top_n: int = 20) -> pd.DataFrame:
    """
    Rank descriptors using five complementary metrics with weighted scoring.

    Metrics:
        1. Pearson correlation (linear, weight=0.10)
        2. Spearman correlation (monotonic, weight=0.20)
        3. F-test (linear model significance, weight=0.10)
        4. Mutual Information (non-linear dependency, weight=0.30)
        5. Random Forest importance (non-linear + interactions, weight=0.30)

    Returns
    -------
    ranked_df : pd.DataFrame
        Descriptors ranked by weighted composite score.
    """
    feature_cols = [c for c in final_df.columns if c != target_col]
    X = final_df[feature_cols].copy()
    y = final_df[target_col].copy()

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    print("\n--- Multi-Metric Screening ---")

    # 1. Pearson correlation
    pearson_corr = final_df.corr(method='pearson')[target_col].drop(target_col)
    print(f"Pearson: top = {pearson_corr.abs().idxmax()} ({pearson_corr.abs().max():.4f})")

    # 2. Spearman correlation
    spearman_corr = final_df.corr(method='spearman')[target_col].drop(target_col)
    print(f"Spearman: top = {spearman_corr.abs().idxmax()} ({spearman_corr.abs().max():.4f})")

    # 3. F-test
    f_test = SelectKBest(score_func=f_regression, k='all')
    f_test.fit(X, y)
    f_scores = pd.Series(f_test.scores_, index=X.columns)
    print(f"F-test: top = {f_scores.idxmax()} ({f_scores.max():.4f})")

    # 4. Mutual Information
    mi = SelectKBest(score_func=mutual_info_regression, k='all')
    mi.fit(X, y)
    mi_scores = pd.Series(mi.scores_, index=X.columns)
    print(f"Mutual Info: top = {mi_scores.idxmax()} ({mi_scores.max():.4f})")

    # 5. Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_scores = pd.Series(rf.feature_importances_, index=X.columns)
    print(f"RF Importance: top = {rf_scores.idxmax()} ({rf_scores.max():.4f})")

    # Composite scoring
    all_scores = pd.DataFrame({
        'Pearson': pearson_corr.abs(),
        'Spearman': spearman_corr.abs(),
        'F_test': f_scores,
        'Mutual_Info': mi_scores,
        'RF_Importance': rf_scores
    })

    # Normalize each metric to [0, 1]
    for col in all_scores.columns:
        col_range = all_scores[col].max() - all_scores[col].min()
        if col_range > 0:
            all_scores[col] = (all_scores[col] - all_scores[col].min()) / col_range
        else:
            all_scores[col] = 0.0

    # Weighted sum
    weights = {'Pearson': 0.10, 'Spearman': 0.20, 'F_test': 0.10,
               'Mutual_Info': 0.30, 'RF_Importance': 0.30}
    all_scores['Weighted_Score'] = sum(all_scores[col] * w for col, w in weights.items())
    all_scores = all_scores.sort_values('Weighted_Score', ascending=False)

    print(f"\nTop {top_n} descriptors by weighted score:")
    print(all_scores.head(top_n)[['Pearson', 'Spearman', 'F_test',
                                   'Mutual_Info', 'RF_Importance', 'Weighted_Score']])

    return all_scores


# ============================================================================
# PART 2: Forward Feature Selection with XGBoost
# ============================================================================

def forward_feature_selection(df: pd.DataFrame, target_col: str,
                               test_size: float = 0.2,
                               random_state: int = 42) -> dict:
    """
    Exhaustive forward feature selection using XGBoost.
    At each step, evaluate all remaining features and select the one
    that minimizes RMSE on the test set.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and target.
    target_col : str
        Name of target column.
    test_size : float
        Fraction of data for testing.
    random_state : int
        Random seed.

    Returns
    -------
    results : dict
        Contains summary_df, all_combinations_df, feature_importance_df,
        best_overall, and execution_time.
    """
    start_time = time.time()

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Clean
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    print(f"\n--- Forward Feature Selection (XGBoost) ---")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    summary_results = []
    all_combinations_results = []
    best_features_so_far = []
    remaining_features = feature_cols.copy()

    for step in range(1, len(feature_cols) + 1):
        best_rmse = float('inf')
        best_r2 = -float('inf')
        best_feature = None

        print(f"\nStep {step}/{len(feature_cols)}: evaluating {len(remaining_features)} candidates...")

        for feature in remaining_features:
            current_combo = best_features_so_far + [feature]

            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X_train[current_combo], y_train)
            y_pred = model.predict(X_test[current_combo])

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            all_combinations_results.append({
                'step': step,
                'feature_added': feature,
                'features': ', '.join(current_combo),
                'num_features': len(current_combo),
                'rmse': rmse,
                'r2': r2
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_r2 = r2
                best_feature = feature

        if best_feature:
            best_features_so_far.append(best_feature)
            remaining_features.remove(best_feature)

            summary_results.append({
                'step': step,
                'feature_added': best_feature,
                'num_features': len(best_features_so_far),
                'features_text': ', '.join(best_features_so_far),
                'rmse': best_rmse,
                'r2': best_r2
            })
            print(f"  Selected: {best_feature} | RMSE={best_rmse:.4f}, R²={best_r2:.4f}")

    # Best overall
    best_overall = min(summary_results, key=lambda x: x['rmse'])
    exec_time = time.time() - start_time

    print(f"\nBest overall: Step {best_overall['step']}")
    print(f"  Features: {best_overall['features_text']}")
    print(f"  RMSE: {best_overall['rmse']:.4f}, R²: {best_overall['r2']:.4f}")
    print(f"  Execution time: {exec_time:.2f}s")

    # DataFrames
    summary_df = pd.DataFrame(summary_results)
    all_combinations_df = pd.DataFrame(all_combinations_results)

    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'selection_step': [next((r['step'] for r in summary_results 
                                  if r['feature_added'] == f), None) 
                             for f in feature_cols]
    }).sort_values('selection_step')

    return {
        'summary': summary_df,
        'all_combinations': all_combinations_df,
        'feature_importance': feature_importance_df,
        'best_overall': best_overall,
        'execution_time': exec_time
    }


def plot_selection_performance(summary_df: pd.DataFrame, output_prefix: str = "XGBoost"):
    """Plot RMSE and R² vs. number of selected features."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(summary_df['step'], summary_df['rmse'], 'o-', color='#1f77b4', markersize=4)
    axes[0].set_xlabel('Number of Features')
    axes[0].set_ylabel('RMSE (°C)')
    axes[0].set_title(f'{output_prefix}: RMSE vs Feature Count')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(summary_df['step'], summary_df['r2'], 'o-', color='#2ca02c', markersize=4)
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('R²')
    axes[1].set_title(f'{output_prefix}: R² vs Feature Count')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_performance_vs_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_prefix}_performance_vs_features.png")


# ============================================================================
# PART 3: Exhaustive Symbolic Descriptor Search
# ============================================================================

def exhaustive_descriptor_search(df: pd.DataFrame, feature_cols: list,
                                  target_col: str,
                                  max_features: int = 6) -> pd.DataFrame:
    """
    Exhaustively search for optimal mathematical expressions (descriptors)
    by combining features through arithmetic operations.

    Operations:
        Unary: sqrt, square, log, exp
        Binary: +, ×, sum of squares
        Ternary to 6-ary: sums, products, and mixed operations

    Features are Min-Max normalized to [0, 1] before operations to ensure
    numerical stability.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with selected top features and target.
    feature_cols : list
        Subset of features to use in expressions.
    target_col : str
        Target variable name.
    max_features : int
        Maximum number of features to combine in one expression.

    Returns
    -------
    results_df : pd.DataFrame
        All valid descriptors ranked by R², with RMSE and expression string.
    """
    start_time = time.time()

    # Normalize features to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Sanitize feature names for expression formatting
    features = [f.replace('-', '_minus_').replace('/', '_div_') for f in feature_cols]

    operations_config = {
        1: [
            ('sqrt', lambda a: np.sqrt(np.clip(a, 1e-10, None)), 'sqrt({0})'),
            ('square', lambda a: a**2, '({0})²'),
            ('log', lambda a: np.log(np.clip(a, 1e-10, None)), 'log({0})'),
            ('exp', lambda a: np.exp(np.clip(a, -50, 50)), 'exp({0})')
        ],
        2: [
            ('+', lambda a, b: a + b, '({0}+{1})'),
            ('×', lambda a, b: a * b, '({0}·{1})'),
            ('sq_sum', lambda a, b: a**2 + b**2, '({0}²+{1}²)')
        ],
        3: [
            ('sum3', lambda a, b, c: a + b + c, '({0}+{1}+{2})'),
            ('prod3', lambda a, b, c: a * b * c, '({0}·{1}·{2})'),
            ('mix3', lambda a, b, c: (a + b) * c, '({0}+{1})·{2}')
        ],
        4: [
            ('sum4', lambda a, b, c, d: a + b + c + d, '({0}+{1}+{2}+{3})'),
            ('prod4', lambda a, b, c, d: a * b * c * d, '({0}·{1}·{2}·{3})'),
            ('mix4_1', lambda a, b, c, d: (a + b) * (c + d), '({0}+{1})·({2}+{3})'),
            ('mix4_2', lambda a, b, c, d: (a * b) + (c * d), '({0}·{1})+({2}·{3})'),
            ('mix4_3', lambda a, b, c, d: (a + b + c) * d, '({0}+{1}+{2})·{3}')
        ],
        5: [
            ('sum5', lambda a, b, c, d, e: a + b + c + d + e, '({0}+{1}+{2}+{3}+{4})'),
            ('prod5', lambda a, b, c, d, e: a * b * c * d * e, '({0}·{1}·{2}·{3}·{4})'),
            ('mix5_1', lambda a, b, c, d, e: (a + b + c) * (d + e), '({0}+{1}+{2})·({3}+{4})'),
            ('mix5_2', lambda a, b, c, d, e: (a * b) + (c * d * e), '({0}·{1})+({2}·{3}·{4})'),
            ('mix5_3', lambda a, b, c, d, e: (a + b) * (c + d + e), '({0}+{1})·({2}+{3}+{4})')
        ],
        6: [
            ('sum6', lambda a, b, c, d, e, f: a + b + c + d + e + f, '({0}+{1}+{2}+{3}+{4}+{5})'),
            ('prod6', lambda a, b, c, d, e, f: a * b * c * d * e * f, '({0}·{1}·{2}·{3}·{4}·{5})'),
            ('mix6_1', lambda a, b, c, d, e, f: (a + b + c) * (d + e + f), '({0}+{1}+{2})·({3}+{4}+{5})'),
            ('mix6_2', lambda a, b, c, d, e, f: (a * b * c) + (d * e * f), '({0}·{1}·{2})+({3}·{4}·{5})'),
            ('mix6_3', lambda a, b, c, d, e, f: (a + b) * (c + d) * (e + f), '({0}+{1})·({2}+{3})·({4}+{5})'),
            ('mix6_4', lambda a, b, c, d, e, f: (a * b) + (c * d) + (e * f), '({0}·{1})+({2}·{3})+({4}·{5})')
        ]
    }

    results = []
    y = df[target_col].values

    for num_feat in range(1, max_features + 1):
        print(f"\nSearching {num_feat}-feature expressions...")
        perms = list(permutations(range(len(features)), num_feat))
        total = len(perms)

        for idx, perm in enumerate(perms):
            if idx % 500 == 0:
                elapsed = time.time() - start_time
                print(f"  {idx}/{total} permutations, elapsed: {elapsed:.1f}s")

            if num_feat not in operations_config:
                continue

            operands = [df_scaled[feature_cols[p]].values for p in perm]
            feat_names = [features[p] for p in perm]

            for op_symbol, op_func, op_format in operations_config[num_feat]:
                try:
                    new_feature = op_func(*operands)

                    # Validity checks
                    valid_mask = np.isfinite(new_feature)
                    if valid_mask.sum() < 3:
                        continue

                    valid_feat = new_feature[valid_mask]
                    valid_y = y[valid_mask]

                    # Correlation
                    r = np.corrcoef(valid_feat, valid_y)[0, 1]
                    if np.isnan(r):
                        continue
                    r2 = r ** 2

                    # Linear regression RMSE
                    a, b = np.polyfit(valid_feat, valid_y, 1)
                    y_pred = a * valid_feat + b
                    rmse = np.sqrt(np.mean((valid_y - y_pred) ** 2))

                    # Expression string
                    descriptor = op_format.format(*feat_names)
                    descriptor = descriptor.replace('_minus_', '-').replace('_div_', '/')

                    results.append({
                        'Descriptor': descriptor,
                        'R2': round(r2, 6),
                        'RMSE': round(rmse, 4),
                        'num_features': num_feat,
                        'operation': op_symbol
                    })
                except Exception:
                    continue

        # Save intermediate results
        if results:
            temp_df = pd.DataFrame(results).sort_values('R2', ascending=False)
            temp_df.to_excel(f'descriptor_search_intermediate_{num_feat}feat.xlsx', index=False)
            print(f"  Saved intermediate: {num_feat}-feature, {len(temp_df)} descriptors")

    # Final deduplication and sorting
    results_df = pd.DataFrame(results).drop_duplicates('Descriptor').sort_values(
        'R2', ascending=False).reset_index(drop=True)

    print(f"\nTotal descriptors found: {len(results_df)}")
    print(f"\nTop 10 descriptors:")
    print(results_df.head(10)[['Descriptor', 'R2', 'RMSE']])

    # Group by feature count
    for n in range(1, max_features + 1):
        group = results_df[results_df['num_features'] == n].head(10)
        if not group.empty:
            print(f"\nTop 10 {n}-feature descriptors:")
            print(group[['Descriptor', 'R2', 'RMSE']])

    results_df.to_excel('exhaustive_descriptor_search_results.xlsx', index=False)
    print(f"\nTotal time: {time.time() - start_time:.1f}s")

    return results_df


# ============================================================================
# 4. Main Pipeline
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "thermal_response_augmented.xlsx"  # WGAN-augmented data
    TARGET_COL = "T"
    FEATURE_COLS = ['CP', 'TC', 'Hr', 'Hv', 'MP', 'YM', 'GS', 'CE', 'EM', 'SVP']
    TOP_N_SCREEN = 20  # Top descriptors from screening for forward selection
    MAX_EXPR_FEATURES = 6  # Max features in symbolic expression search

    # Step 1: Load data
    print("=" * 60)
    print("SISSO-like Descriptor Pipeline")
    print("=" * 60)

    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded {df.shape[0]} samples, {df.shape[1]} columns")

    # Step 2: Feature engineering & screening
    final_df = build_transformed_features(df, FEATURE_COLS, TARGET_COL)
    screened_scores = multi_metric_screening(final_df, TARGET_COL, top_n=TOP_N_SCREEN)

    # Save screening results
    screened_scores.to_excel('descriptor_screening_results.xlsx')
    print("\nSaved: descriptor_screening_results.xlsx")

    # Step 3: Forward feature selection on top screened descriptors
    top_descriptors = screened_scores.head(TOP_N_SCREEN).index.tolist()
    selection_df = final_df[top_descriptors + [TARGET_COL]]

    ffs_results = forward_feature_selection(selection_df, TARGET_COL)
    plot_selection_performance(ffs_results['summary'])

    # Save FFS results
    with pd.ExcelWriter('forward_feature_selection_results.xlsx') as writer:
        ffs_results['summary'].to_excel(writer, sheet_name='summary', index=False)
        ffs_results['all_combinations'].to_excel(writer, sheet_name='all_combinations', index=False)
        ffs_results['feature_importance'].to_excel(writer, sheet_name='feature_importance', index=False)
    print("\nSaved: forward_feature_selection_results.xlsx")

    # Step 4: Exhaustive symbolic descriptor search on best features
    best_n = ffs_results['best_overall']['num_features']
    best_features = ffs_results['best_overall']['features_text'].split(', ')

    print(f"\n--- Exhaustive Search on Best {best_n} Features ---")
    print(f"Features: {best_features}")

    search_df = df[best_features + [TARGET_COL]]
    exhaustive_descriptor_search(search_df, best_features, TARGET_COL,
                                  max_features=min(MAX_EXPR_FEATURES, best_n))
