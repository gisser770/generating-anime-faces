"""
Schema for config file
"""

CFG_SCHEMA = {
    'main': {
        'experiment_name_prefix': str,
        'seed': int,
        'num_workers': int,
        'parallel': bool,
        'gpus_to_use': str,
        'trains': bool,
        'train_size': float,
        'auto_encoder_latent_dim': int,
        'paths': {
            'train': {
                'path': str,
                'number_of_samples': int,
            },
            'validation': str,
            'logs': str,
        },
    },
    'train': {
        'num_epochs': int,
        'grad_clip': float,
        'dropout': float,
        'num_hid': int,
        'batch_size': int,
        'save_model': bool,
        'GaussianNLLLoss_var': float,
        'lr': {
            'lr_value': float,
            'lr_decay': int,
            'lr_gamma': float,
            'lr_step_size': int,
        },
    },
}
