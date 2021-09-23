'''
Author: jojo
Date: 2021-08-09 03:47:36
LastEditors: jojo
LastEditTime: 2021-09-13 07:57:18
FilePath: /210610338/utils/config.py
'''
import os
import yaml
from .const import CONFIG_PATH, LOG_PATH

def get_config(config_path = CONFIG_PATH):
    # 打开yaml文件
    with open(config_path,encoding="UTF-8") as fs:
        datas = yaml.load(fs,Loader=yaml.FullLoader)  #添加后就不警告了
        return datas
    
config = get_config()

def get_eval_config(log_name):
    if not str(log_name).endswith('.log'):
        log_name = f'{log_name}.log'
    log_path = os.path.join(LOG_PATH, log_name)
    if config['device']=='Ascend':
        import moxing as mox
        obs_log = config['obs_log']
        obs_log_path = obs_log + log_name
        mox.file.copy_parallel(obs_log_path, log_path)
    datas = get_config(log_path)
    return datas



if __name__=='__main__':
    datas = get_config()
    print(datas)