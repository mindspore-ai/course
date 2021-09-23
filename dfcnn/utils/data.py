'''
Author: jojo
Date: 2021-07-10 04:51:50
LastEditors: jojo
LastEditTime: 2021-08-27 08:58:52
FilePath: /210610338/utils/data.py
Description: 数据处理工具
reference: https://blog.csdn.net/chinatelecom08/article/details/85013535
'''
import os
import mindspore.context as context
import numpy as np
from utils import compute_time_freq_images, divmod
from .const import DATA_PATH
import mindspore.dataset as ds


class thchs30(object):
    def __init__(self, dataset_path, batch_size, div = 8, stage='train', wav_max_len=1600, lab_max_len=50):
      """初始化

        Args:
            dataset_path (str): 数据集根目录
            div (int, optional): 整除除数，根据模型的特征，前向传播后max_time维度缩小了8倍 . Defaults to 8.
            stage (str, optional): 数据集所属阶段: {train,test,dev}. Defaults to 'train'.
            wav_max_len (int, optional): 语谱图最长长度. Defaults to 1640.
            lab_max_len (int, optional): 标签数组最长长度. Defaults to 50.
        """
      self.dataset_path = dataset_path
      self.label_list_path = os.path.join(dataset_path, 'label.txt')
      self.stage = stage # train、test、eval阶段
      self.wav_max_len = wav_max_len
      self.lab_max_len = lab_max_len
      self.div = div
      self.batch_size = batch_size


      # 拿到音频文件、对应标注文件列表
      self.wav_list, self.label_lst = self.get_thchs_data(dir=self.stage)

      self.dataset_size = len(self.wav_list) // self.batch_size
      
      # 生成标签数据
      self.label_data = self.generate_thchs_label_data(label_lst=self.label_lst,dir=self.stage)

      # 建立整个数据集的标签2id的索引
      if not os.path.exists(self.label_list_path):
          # 生成标签列表文件
          self.generate_label_file()
          
      self.idx2label, self.label2idx = self.build_vocab()

    def get_vocab(self):
        return self.idx2label, self.label2idx

    def generate_label_file(self):
        _, all_label_lst = self.get_thchs_data(dir='data')
        all_label_data = self.generate_thchs_label_data(label_lst=all_label_lst,dir='data')
        label_list = []
        for label_line in all_label_data:
            lab_lst = label_line.split(' ')
            for lab in lab_lst:
                # 去重
                if not lab in label_list:
                    label_list.append(lab)

        with open(self.label_list_path, 'w') as f:
            for label in label_list:
                line = f'{label}\n'
                f.write(line)
            
            # 空白字符串
            f.write('_')

     
    def get_thchs_data(self, dir):
        """从thchs30数据源中获取音频文件以及标注文件的列表

        Args:
            dir (str): 数据集路径:{'train','dev','test'}

        Returns:
            tuple: (标注列表，音频列表)
        """
        # train_file = source_file + '/data'
        data_file = os.path.join(self.dataset_path, dir)
        label_lst = []
        wav_lst = []
        for root, dirs, files in os.walk(data_file):
            for file in files:
                if file.endswith('.wav') or file.endswith('.WAV'):
                    wav_file = os.sep.join([root, file])
                    label_file = wav_file + '.trn'
                    wav_lst.append(wav_file)
                    label_lst.append(label_file)
        
        # 检查音频、标签对应相同
        self.check(
            label_lst=label_lst,
            wav_lst=wav_lst
        )
        return wav_lst, label_lst

    def check(self,label_lst,wav_lst):
        """检查音频、标签对应相同

        Args:
            label_lst (list): 标签列表
            wav_lst (list): 音频文件列表
        """
        label_len = len(label_lst)
        wav_len = len(wav_lst)
        
        assert label_len==wav_len,"音频文件与音频标签数量不匹配，请检查数据集是否完整"

        for i in range(label_len):
            wavname = (wav_lst[i].split('/')[-1]).split('.')[0]
            labelname = (label_lst[i].split('/')[-1]).split('.')[0]
            assert wavname == labelname,f"音频文件: {wavname}与 标签文件: {labelname} 不匹配，请检查"

    def read_thchs_label(self,label_file,dir):
        """读取thchs30的标签文件

        Args:
            label_file (str): 标签路径

        Returns:
            str: 标签行
        """
        with open(label_file, 'r', encoding='utf8') as f:
            data = f.readlines()
            if dir in ['train','test','dev']:
                # 获取真实的路径
                root = label_file.split(self.stage)[0]
                real_path = os.path.join(root,self.stage,data[0].strip())
                with open(real_path, 'r', encoding='utf8') as f2:
                     data2 = f2.readlines()
                     return data2[1]
            else:
                return data[1]

        
    
    def generate_thchs_label_data(self,label_lst,dir):
        """生成标签列表

        Args:
            label_lst (list): 标签文件列表

        Returns:
            list: 标签字符串列表, 形如：
                ['a2 a4 ....'],
                ['b2 b4 ....'],
        """
        label_data = []
        for label_file in label_lst:
            pny = self.read_thchs_label(label_file=label_file,dir=dir)
            label_data.append(pny.strip('\n'))
        return label_data

    def build_vocab(self):
        """建立索引，id映射到标签，标签映射到id

        Args:
            label_data (list): 标签列表

        Returns:
            tuple: 
                idx2label (dict): 键为id，值为label
                label2idx (dict): 键为label，值为id
        """
        label_list = self.get_label_list()
        idx2label = {}
        label2idx = {}

                    
        for idx,label in enumerate(label_list):
            idx2label[idx] = label
            label2idx[label] = idx
            
        return idx2label,label2idx

    def get_label_list(self):
        label_list = []
        # 打开标签表
        with open(self.label_list_path, 'r') as f:
            for line in f.readlines():
                label = line.strip()
                label_list.append(label)

            return label_list
            

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self,index):
        """随机索引

        Args:
            index (int): 下标
        """
        img_batch = []
        label_batch = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            wav_path, label_path = self.wav_list[i], self.label_lst[i]

            # 处理音频
            # 生成语谱图
            fq_img = compute_time_freq_images(wav_path)
            fq_img = divmod(fq_img,self.div)
            # fq_img = self.wav_pad(fq_img)
            
            # 处理标签
            # 标签转id
            label = self.read_thchs_label(label_file=label_path,dir=self.stage)
            label, len = self.line2id(label, self.label2idx)
                    
            # label = self.lab_pad(label)

            img_batch.append(fq_img)
            label_batch.append(label)
            
        img_batch, wav_max_len = self.wav_pad(img_batch)
        # label_batch, lab_len = self.lab_pad(label_batch)
        label_batch, lab_len = self.encode(label_batch)
    
        label_indices = []
        for i, _ in enumerate(lab_len):
            for j in range(lab_len[i]):
                label_indices.append((i, j))
        
        label_indices = np.array(label_indices, dtype=np.int64)
        sequence_length = np.array([wav_max_len // self.div] * self.batch_size, dtype=np.int32)

        # img_batch = np.array(img_batch, dtype= np.float32)
        # label_batch = np.array(label_batch, dtype= np.int32)
        
        return img_batch, label_indices, label_batch, sequence_length, lab_len
    
    def line2id(self,line,label2idx):
        """将单行语音的标签映射为id

        Args:
            line (list): 单行语音标签
            label2idx (dict): 标签->id的映射字典

        Returns:
            list: 映射的id列表
        """
        label =  [label2idx[pny.strip()] for pny in line.split(' ')]
        return label, len(label)
    
    def wav_pad(self,wav_data_lst):
        """时频图数组按照 最长序列 进行补齐padding，传入的数据shape: [batch_size, max_time, 200]
           以及生成ctc需要获得的信息：输入序列的长度列表
        """
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // self.div for leng in wav_lens])
        # new_wav_data_lst = np.zeros((len(wav_data_lst), 1, wav_max_len, 200), dtype=np.float32) 出问题则删掉下面那句,恢复这句
        new_wav_data_lst = np.zeros((len(wav_data_lst), 1, self.wav_max_len, 200), dtype=np.float32)
        for i in range(len(wav_data_lst)):
            if len(wav_data_lst[i])<=self.wav_max_len:
                new_wav_data_lst[i, 0, 0 :wav_data_lst[i].shape[0], :] = wav_data_lst[i]
            else:
                new_wav_data_lst[i, 0, 0 :wav_data_lst[i].shape[0], :] = wav_data_lst[i][: self.wav_max_len]

        return new_wav_data_lst, wav_max_len

    def encode(self, label):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        lab_lst = []
        length = [len(s) for s in label]
        for labs in label:
            for lab in labs:
                lab_lst.append(lab)

        return np.array(lab_lst, dtype=np.int32), np.array(length)
        
    def lab_pad(self, label_data_lst):
        """标签数组按照 最长序列 进行补齐 padding
        """
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len), dtype=np.int32)
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens
        
def get_dataset(dataset_path =DATA_PATH, stage = 'train', div=8, batch_size=8, test_dev_batch_size = 8, num_parallel_workers=1):
    """获取训练集、测试机、验证集、以及id->标签的映射词典、标签->id的映射词典

    Args:
        dataset_path (str, optional): 数据集路径. Defaults to '/home/jojo/PythonProjects/speech_data/data_thchs30'.
        div (int, optional): 需被整除的除数. Defaults to 1.
        batch_size (int, optional): batch大小. Defaults to 64.
        num_parallel_workers (int, optional): 线程数. Defaults to 4.

    Returns:
        tuple: 训练集、测试机、验证集、以及id->标签的映射词典、标签->id的映射词典
    """
    if stage == 'train':
        train_generator = thchs30(dataset_path=dataset_path, batch_size= batch_size,
                                div=div,stage='train')

        train_dataset = ds.GeneratorDataset(source=train_generator,
                column_names=['img', 'label_indices', 'label_batch', 'sequence_length', 'lab_len'], 
                shuffle=True, 
                num_parallel_workers=num_parallel_workers)

        # vocab
        idx2label ,label2idx = train_generator.get_vocab()

        return train_dataset, idx2label ,label2idx

    elif stage == 'test':
        test_generator = thchs30(dataset_path=dataset_path, batch_size= test_dev_batch_size,
                                div=div,
                                stage='test')

        test_dataset = ds.GeneratorDataset(source=test_generator,
                                        column_names=['img', 'label_indices', 'label_batch', 'sequence_length', 'lab_len'], 
                                        shuffle=False, 
                                        num_parallel_workers=num_parallel_workers)

        # vocab
        idx2label ,label2idx = test_generator.get_vocab()

        return test_dataset, idx2label ,label2idx
    
    elif stage == 'dev':
        dev_generator = thchs30(dataset_path=dataset_path, batch_size= test_dev_batch_size,
                                div=div,
                                stage='dev')
        
        dev_dataset = ds.GeneratorDataset(source=dev_generator,
                                            column_names=['img', 'label_indices', 'label_batch', 'sequence_length', 'lab_len'], 
                                            shuffle=True, 
                                            num_parallel_workers=num_parallel_workers)

        # vocab
        idx2label ,label2idx = dev_generator.get_vocab()

        return dev_dataset, idx2label ,label2idx


if __name__=='__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)

    test_dataset,i2l,l2i = get_dataset(num_parallel_workers=1, batch_size=4)
    # max_wav_len = 0
    # max_lab_len = 0
    # # for train_,test,dev in zip(train_dataset.create_dict_iterator(), test_dataset.create_dict_iterator(), dev_dataset.create_dict_iterator()):
    # for train_ in train_dataset.create_dict_iterator():
    #     print('{}'.format(train_["img"].shape), '{}'.format(train_["label_indices"].shape))
    #     print('{}'.format(train_["label_batch"].shape), '{}'.format(train_["sequence_length"].shape))
    count = 0
    for test in test_dataset.create_dict_iterator():
        count += 1
        print('{}'.format(test["img"].shape), '{}'.format(test["label_indices"].shape))
        print('{}'.format(test["label_batch"].shape), '{}'.format(test["sequence_length"].shape))

    print(count)