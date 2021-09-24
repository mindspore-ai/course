"""
logger
"""
import os
import time
import datetime

from .const import LOG_PATH

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)


class logger:
    def __init__(self, config: dict, start_date: str):
        self.config = config
        self.start_date = start_date
        self.log_name = start_date
        self.log_path = os.path.join(LOG_PATH, f'{start_date}.log')
        self.start_time = time.time()
        self.best_acc = None
        self.best_ckpt = None
        self.best_time = None
        self.logging()

    def logging(self, latest_ckpt_file_name=None):
        with open(self.log_path, 'w') as f:
            line = f'log_name: {self.log_name}\n'
            f.write(line)
            for key, item in self.config.items():
                line = f'{key}: {item}\n'
                f.write(line)

            if not latest_ckpt_file_name == None:
                msg = f'# the latest model is saved to {latest_ckpt_file_name}\nlatest_model: {latest_ckpt_file_name}\n'
                f.write(msg)

            if not self.best_acc == None and not self.best_time == None:
                cost_time_for_best_acc = datetime.timedelta(seconds=self.best_time - self.start_time)
                msg = f'# the best acc now is {self.best_acc}, costing {cost_time_for_best_acc}\nbest_acc: {self.best_acc}\n'
                f.write(msg)

            if not self.best_ckpt == None:
                msg = f'#　the responding model path for best acc is save to {self.best_ckpt}\nbest_ckpt: {self.best_ckpt}\n'
                f.write(msg)

            log_time = time.time()
            cost_time = datetime.timedelta(seconds=log_time - self.start_time)

            msg = f'# has been training for {cost_time}'
            f.write(msg)

    def update_acc_ckpt(self, best_acc, best_ckpt):
        self.best_acc = best_acc
        self.best_ckpt = best_ckpt
        self.best_time = time.time()
        self.logging()
