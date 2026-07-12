#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd  
from sklearn.model_selection import train_test_split  
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')


class PhysicsGradientGuidedAL:
    """Physics-gradient-guided active learning framework (with hard constraints)"""
    
    def __init__(self, catboost_params, n_bootstrap=30, physics_dir_weight=0.2):
        """
        Parameters:
            catboost_params: CatBoost hyperparameter dictionary
            n_bootstrap: Number of bootstrap samples
            physics_dir_weight: Guidance weight of physics gradient direction (0-1)
        """
        self.catboost_params = catboost_params
        self.n_bootstrap = n_bootstrap
        self.physics_dir_weight = physics_dir_weight
        self.models = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = ['TC', 'Hr', 'Hv', 'GS', 'CE', 'EM', 'SVP']
        self.best_x_raw = None  
        
    def _compute_physical_gradient(self, x):
        """
        Compute partial derivatives of the explicit equation
        Return physics gradient vector (sensitivity)
        """
        eps = 1e-6
        TC, Hr, Hv, GS, CE, EM, SVP = np.maximum(x, eps)
        
        A = CE**(1/3) / (SVP**3 * Hv**(1/3))
        B = CE**2 / (EM**3 * TC**(1/2))
        Hr_term = Hr**(1/3)
        
        dT_dTC = -0.5 * Hr_term * CE**2 / (EM**3 * TC**1.5)
        dT_dHr = (1/3) * Hr**(-2/3) * (A + B)
        dT_dHv = -(1/3) * Hr_term * CE**(1/3) / (SVP**3 * Hv**(4/3))
        dT_dGS = 0.0  
        dT_dCE = Hr_term * ((1/3)*CE**(-2/3)/(SVP**3 * Hv**(1/3)) + 2*CE/(EM**3 * TC**(0.5)))
        dT_dEM = -3.0 * Hr_term * CE**2 / (EM**4 * TC**(0.5))
        dT_dSVP = -3.0 * Hr_term * CE**(1/3) / (SVP**4 * Hv**(1/3))
        
        gradient = np.array([dT_dTC, dT_dHr, dT_dHv, dT_dGS, dT_dCE, dT_dEM, dT_dSVP])
        return gradient

    def train_ensemble(self, X_train, y_train):
        """Train ensemble surrogate models and determine the current best anchor"""
        self.models = []
        n_samples = len(X_train)
        
        best_idx = np.argmin(y_train)
        self.best_x_raw = X_train[best_idx]
        
        X_scaled = self.scaler_X.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        print(f"Training {self.n_bootstrap} bootstrap surrogate models")
        for i in range(self.n_bootstrap):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            model = CatBoostRegressor(**self.catboost_params, verbose=0)
            model.fit(X_scaled[indices], y_scaled[indices])
            self.models.append(model)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{self.n_bootstrap}")
                
    def get_physics_directional_score(self, X_pool):
        """Compute cosine similarity between candidate formulations and the physics steepest-descent direction"""
        raw_gradient = self._compute_physical_gradient(self.best_x_raw)
        scaled_gradient = raw_gradient * self.scaler_X.scale_
        
        ideal_direction = -scaled_gradient  # Search for low T, take negative gradient direction
        ideal_norm = np.linalg.norm(ideal_direction) + 1e-10
        
        Z_best = self.scaler_X.transform(self.best_x_raw.reshape(1, -1))
        Z_pool = self.scaler_X.transform(X_pool)
        search_vectors = Z_pool - Z_best
        
        directional_scores = []
        for vec in search_vectors:
            vec_norm = np.linalg.norm(vec) + 1e-10
            cos_sim = np.dot(vec, ideal_direction) / (vec_norm * ideal_norm)
            directional_scores.append(cos_sim)
            
        return np.array(directional_scores)

    def predict_with_uncertainty(self, X_pool):
        """Surrogate model prediction of mean and uncertainty"""
        X_scaled = self.scaler_X.transform(X_pool)
        predictions = np.array([model.predict(X_scaled) for model in self.models])
        
        pred_scaled = np.mean(predictions, axis=0)
        std_scaled = np.std(predictions, axis=0)
        
        mean_pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        std_pred = std_scaled * self.scaler_y.scale_[0]
        
        return mean_pred, std_pred

    def select_next_samples(self, X_pool, n_samples=4, diversity_weight=0.3):
        """Execute active learning sample selection (with hard constraint of physics direction > 0)"""
        X_pool_array = X_pool.values if isinstance(X_pool, pd.DataFrame) else X_pool
        
        # 1. Surrogate model prediction
        mean_pred, std_pred = self.predict_with_uncertainty(X_pool_array)
        kappa = 2.0
        ml_ucb = -(mean_pred - kappa * std_pred)
        
        # 2. Physics gradient direction score (cosine similarity)
        physics_dir_scores = self.get_physics_directional_score(X_pool_array)
        
        # 3. Rank-space fusion
        ml_ranks = rankdata(ml_ucb)
        phys_ranks = rankdata(physics_dir_scores)
        
        ml_ranks_norm = (ml_ranks - ml_ranks.min()) / (ml_ranks.max() - ml_ranks.min() + 1e-10)
        phys_ranks_norm = (phys_ranks - phys_ranks.min()) / (phys_ranks.max() - phys_ranks.min() + 1e-10)
        
        combined_scores = (1 - self.physics_dir_weight) * ml_ranks_norm + self.physics_dir_weight * phys_ranks_norm
        
        # =======================================================
        # Core improvement: introduce hard constraint of physics optimization alignment > 0
        # =======================================================
        valid_mask = physics_dir_scores > 0
        
        valid_count = np.sum(valid_mask)
        if valid_count < n_samples:
            print(f"\n[ Warning] Only {valid_count} candidate samples with physics optimization alignment > 0 remain, insufficient for {n_samples}!")
                        
        # Forcefully remove candidate scores that do not satisfy physics positive guidance (≤0)
        combined_scores[~valid_mask] = -np.inf
        
        # 4. Sequential recommendation (considering diversity)
        selected_indices = []
        selected_features = []
        current_scores = combined_scores.copy()
        
        for i in range(n_samples):
            # Exception handling: if all samples satisfying hard constraint are selected, fallback processing
            if np.max(current_scores) == -np.inf:
                fallback_scores = (1 - self.physics_dir_weight) * ml_ranks_norm + self.physics_dir_weight * phys_ranks_norm
                fallback_scores[selected_indices] = -np.inf  # Exclude already selected
                current_scores = fallback_scores
                
            if i == 0:
                best_idx = np.argmax(current_scores)
            else:
                selected_array = np.array(selected_features)
                Z_pool = self.scaler_X.transform(X_pool_array)
                Z_selected = self.scaler_X.transform(selected_array)
                min_dist = np.min([np.linalg.norm(Z_pool - sel, axis=1) for sel in Z_selected], axis=0)
                
                min_dist_norm = (min_dist - min_dist.min()) / (min_dist.max() - min_dist.min() + 1e-10)
                
                # Combine diversity and surrogate model + physics guidance scores
                div_scores = (1 - diversity_weight) * current_scores + diversity_weight * min_dist_norm
                
                # Strictly maintain original mask (samples not satisfying hard constraint remain -inf)
                div_scores[current_scores == -np.inf] = -np.inf
                div_scores[selected_indices] = -np.inf
                
                best_idx = np.argmax(div_scores)
                
            selected_indices.append(best_idx)
            selected_features.append(X_pool_array[best_idx])
            current_scores[best_idx] = -np.inf  # Mark as selected
            
        return selected_indices, mean_pred, std_pred, physics_dir_scores


def run_gradient_guided_al(train_file, pool_file, n_recommendations=4):
    print("="*85)
    print(" Advanced Active Learning Framework Based on Physics-Gradient-Direction Hard Constraint (Alignment > 0)")
    print("="*85)
    
    # 1. Load data
    train_data = pd.read_excel(train_file)
    pool_data = pd.read_excel(pool_file)
    feature_cols = ['TC', 'Hr', 'Hv', 'GS', 'CE', 'EM', 'SVP']
    X_train = train_data[feature_cols].values
    y_train = train_data['T'].values
    X_pool = pool_data[feature_cols]
    
    print(f"[Data Status] Training set: {len(X_train)} samples | Pool: {len(X_pool)} samples")
    print(f"Current training set best (lowest) T value: {y_train.min():.2f}")
    
    # 2. Initialize framework
    catboost_params = {
        'random_strength': 3, 'learning_rate': 0.1, 'l2_leaf_reg': 7,
        'iterations': 300, 'depth': 7, 'colsample_bylevel': 0.8,
        'bagging_temperature': 1, 'random_seed': 42
    }
    
    al_framework = PhysicsGradientGuidedAL(catboost_params, n_bootstrap=50, physics_dir_weight=0.2)
    
    # 3. Train
    al_framework.train_ensemble(X_train, y_train)
    
    # 4. Recommend candidate samples
    selected_indices, mean_pred, std_pred, phys_scores = al_framework.select_next_samples(X_pool, n_samples=n_recommendations)
    
    # 5. Output
    print("\n" + "-"*85)
    print(" Final Recommended Formulations (all recommendations satisfy: exploration direction consistent with physics cooling mechanism)")
    print("-" * 85)
    
    recommendations = pool_data.iloc[selected_indices].copy()
    recommendations['ML_Predicted_T'] = mean_pred[selected_indices]
    recommendations['Uncertainty'] = std_pred[selected_indices]
    recommendations['Physics_Optimization_Alignment'] = phys_scores[selected_indices]
    
    for i, idx in enumerate(selected_indices):
        is_valid = "Satisfies hard constraint" if phys_scores[idx] > 0 else "Triggers fallback degradation"
        print(f"\n[Recommended Formulation #{i+1}] (Original index in pool: {idx})")
        print(f"  Surrogate model predicted T : {mean_pred[idx]:.2f} ± {std_pred[idx]:.2f}")
        print(f"  Physics optimization alignment: {phys_scores[idx]:.4f} ({is_valid})")
        print(f"  Formulation parameters: ", end="")
        for feat in feature_cols:
            print(f"{feat}={pool_data.iloc[idx][feat]:.2f}  ", end="")
        print()
        
    recommendations.to_excel('Physics_Gradient_Constrained_Recommendations.xlsx', index=False)
    print("\n=> Recommendation results saved to: Physics_Gradient_Constrained_Recommendations.xlsx")
    print("="*85)

if __name__ == "__main__":
    run_gradient_guided_al(
        train_file="Active_Learning_Training_Data.xlsx", 
        pool_file="Active_Learning_Formulation_Pool.xlsx", 
        n_recommendations=4
    )

