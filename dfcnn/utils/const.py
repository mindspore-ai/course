"""
constant value
"""
import os

current_file_dir = os.path.abspath(os.path.dirname(__file__))
main_dir = os.path.abspath(os.path.join(current_file_dir, '..'))

CONFIG_PATH = os.path.join(main_dir, 'config.yaml')
LOG_PATH = os.path.join(main_dir, 'log')
SUMMARY_DIR = os.path.join(main_dir, 'summary_dir')

DATA_ROOT_PATH = os.path.join(main_dir, 'dataset')
DATA_PATH = os.path.join(DATA_ROOT_PATH, 'data_thchs30')

MODEL_PATH = os.path.join(main_dir, 'saved_model')
BEST_MODEL_PATH = os.path.join(main_dir, 'saved_model', 'best_model')

if __name__ == '__main__':
    print(current_file_dir)
    print(main_dir)
    print(CONFIG_PATH)
