"""
network config
"""
from easydict import EasyDict as edict

# LSTM CONFIG
lstm_cfg = edict({
    'num_classes': 2,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'num_epochs': 10,
    'batch_size': 32,
    'embed_size': 200,
    'num_hiddens': 100,
    'num_layers': 1,
    'bidirectional': False,
    'save_checkpoint_steps': 390*5,
    'keep_checkpoint_max': 10
})
