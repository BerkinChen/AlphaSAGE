"""
Configuration file for grid search parameters.
Modify this file to customize the parameter search space.
"""

# Base configuration (fixed parameters from run.sh)
BASE_CONFIG = {
    'seed': 0,
    'instrument': 'csi300',
    'pool_capacity': 50,
    'log_freq': 500,
    'update_freq': 64,
    'n_episodes': 10000,  # Will be reduced to 1000 in quick mode
    'encoder_type': 'gnn',
    'weight_decay_type': 'linear',
    'final_weight_ratio': 0.0
}

# Full parameter grid for comprehensive search
FULL_PARAM_GRID = {
    'entropy_coef': [0.0, 0.01],
    'entropy_temperature': [1.0],
    'mask_dropout_prob': [1.0],
    'ssl_weight': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'nov_weight': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}

# Quick test grid for development/debugging
QUICK_PARAM_GRID = {
    'entropy_coef': [0.0, 0.01],
    'entropy_temperature': [1.0],
    'mask_dropout_prob': [0.7, 1.0],
    'ssl_weight': [0.0, 0.5],
    'nov_weight': [0.0, 0.3]
}

# Custom focused grid - modify this for targeted search
FOCUSED_PARAM_GRID = {
    'entropy_coef': [0.0, 0.01, 0.05],
    'entropy_temperature': [1.0, 2.0],
    'mask_dropout_prob': [0.7, 1.0],
    'ssl_weight': [0.0, 0.3, 0.7],
    'nov_weight': [0.1, 0.3, 0.5]
}

# Parameter descriptions for reference
PARAM_DESCRIPTIONS = {
    'entropy_coef': 'Coefficient for entropy regularization in GFN loss',
    'entropy_temperature': 'Temperature parameter for entropy calculation',
    'mask_dropout_prob': 'Probability of masking valid actions based on expression length',
    'ssl_weight': 'Initial weight for SSL (self-supervised learning) reward',
    'nov_weight': 'Initial weight for novelty reward'
}
