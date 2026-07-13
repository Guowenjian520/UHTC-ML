# UHTC-ML: Interpretable Machine Learning for Ultra-High-Temperature Ceramics

This repository contains the interpretable machine-learning framework for designing ultra-high-temperature ceramics (UHTCs) with low thermal response temperatures, as described in the manuscript *"Self-organized geometric blackbody microstructures for radiative cooling in ultra-high-temperature ceramics designed via interpretable machine learning"* (Nature Communications).

The framework integrates:
- **WGAN-based data augmentation** for expanding limited experimental datasets
- **SISSO-like directional descriptor construction** for physics-consistent feature engineering
- **High-throughput virtual screening** using the CBR (Case-Based Reasoning) algorithm
- **Physics-gradient-guided active learning** for efficient experimental navigation

---

## 1. System Requirements

### Software Dependencies
- **Python** &gt;= 3.8
- **PyTorch** &gt;= 1.9.0
- **scikit-learn** &gt;= 0.24.0
- **XGBoost** &gt;= 1.4.0
- **pandas** &gt;= 1.3.0
- **numpy** &gt;= 1.21.0
- **matplotlib** &gt;= 3.4.0
- **seaborn** &gt;= 0.11.0
- **umap-learn** &gt;= 0.5.0

### Operating Systems
Tested on:
- Ubuntu 20.04 LTS
- Windows 10/11
- macOS 12+

### Hardware Requirements
- **No non-standard hardware is required.** The code runs on a standard desktop CPU.
- **GPU is optional** and only recommended for accelerating WGAN training. All other modules (descriptor construction, CBR prediction, active learning) run efficiently on CPU.
