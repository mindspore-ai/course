'''
Author: jojo
Date: 2021-07-10 04:52:29
LastEditors: jojo
LastEditTime: 2021-08-28 11:21:50
FilePath: /210610338/utils/utils.py
'''
from mindspore.common.tensor import Tensor
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

# from .decoder import ctc_beam_decode, beam_search_decode, softmax
# 音频文件路径
filepath = '/home/jojo/PythonProjects/speech_data/data_thchs30/data/A11_0.wav'

img_path = '/home/jojo/PythonProjects/DFCNN-CTC/summer/210610338/fq2img'

def compute_time_freq_images(filepath):
    """生成音频时频图

    Args:
        filepath (str): 文件地址

    Returns:
        numpy.array: 视频特征图
    """
    # -------------- 1.读取音频文件 ------------
    fs, wavsignal = wav.read(filepath) # wavsignal 即为频谱图
    
    # 转换为numpy数组
    wav_arr = np.array(wavsignal)

    # 频谱长度
    # wav_length = len(wavsignal)
    wav_length = wav_arr.shape[0]
    
    # -------------  2.汉明窗 ----------------
    x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))

    # 采样点（s） = fs
    # 采样点（ms）= fs / 1000
    # 采样点（帧）= fs / 1000 * 帧长

    # ------------- 3.对数据分帧 --------------
    # 帧长： 25ms
    time_window = 25

    # 帧移： 10ms
    window_length = fs // 1000 * time_window

    # -------------- 4.分帧加窗 ------------------
    
    # 最终生成的窗数
    window_end = int(wav_length/fs*1000 - time_window) // 10
    
    # 最终输出结果
    data_input = np.zeros((window_end,window_length//2),dtype=np.float)
    
    data_line = np.zeros((1, window_end), dtype = np.float)
    
    # 分帧
    for i in range(0, window_end):
        p_start = i * 160
        p_end = p_start + window_length
        data_line = wav_arr[p_start:p_end]	
        data_line = data_line * w # 加窗
        
        # -------- 5.傅里叶变换：生成时频图 -------------
        data_line = np.abs(fft(data_line)) / wav_length
        data_input[i]=data_line[0:window_length//2] # 设置为window_length除以2的值（即200）是取一半数据，因为是对称的

    # 取对数，求db
    data_input = np.log(data_input + 1)

    return data_input

def divmod(array,div=8):
    """整除处理

    Args:
        array (numpy.array): 数组
        div (int, optional): 欲被整除数. Defaults to 8.

    Returns:
        numpy.array: 整除后的数
    """
    return array[:array.shape[0]//div*div, :]

class CTCLabelConverter():
    """ Convert between text-label and text-index 
        https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/cnnctc/src/util.py
    """

    def __init__(self, label2idx:dict, idx2label:dict, batch_size):
        """ctc解码工具

        Args:
            label2idx (dict): 标签->id的映射,最后一位应为空字符
            idx2label (dict): id->标签的映射,最后一位应为空字符
            batch_size (int): batch的大小
        """
        self.batch_size = batch_size
        self.label2idx = label2idx
        self.idx2label = idx2label

        self.vocal_list = []

        for key,item in idx2label.items():
            if key == len(idx2label)-1:
                continue
            self.vocal_list.append(item)


    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.label2idx[char] for char in text]

        return np.array(text), np.array(length)

    def decode(self, text_index, length, merge_repeat = True):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                    if not t[i]==self.label2idx['_']: # removing repeated characters and blank.
                        if merge_repeat:
                            # 去掉重复项
                            if i>0 and not t[i - 1] == t[i]:
                                char_list.append(self.idx2label[t[i]])
                        else:
                            char_list.append(self.idx2label[t[i]])
            # text = ''.join(char_list)

            # texts.append(text)
            texts.append(char_list)
            index += l
        return texts

    
    def decode_label(self, label, label_len):
        label = np.squeeze(label.asnumpy())
        # label_size = np.array([label.shape[1]] * self.batch_size)
        try:
            label_size = np.array([len.asnumpy() for len in label_len])
        except:
            label_size = np.array([len for len in label_len])
        label = np.reshape(label, [-1])
        label_str = self.decode(label, label_size, merge_repeat=False)
        return label_str
    
    def ctc_decoder(self,model_predicts):
        """模型输出值

        Args:
            model_predict (mindspore.Tensor): 模型输出值

        Returns:
            list: 解码之后的标签字符串列表，长度为batch_size
        """
        model_predicts = np.squeeze(model_predicts.asnumpy())

        # 去掉每个time最后一个维度
        # model_predicts = np.delete(model_predicts, obj=-1, axis=-1)
        preds_size = np.array([model_predicts.shape[1]] * self.batch_size)
        preds_index = np.argmax(model_predicts, 2)
        preds_index = np.reshape(preds_index, [-1])
        preds_str = self.decode(preds_index, preds_size)

        return preds_str
    
import difflib

def GetEditDistance(str1, str2):
	leven_cost = 0
	s = difflib.SequenceMatcher(None, str1, str2)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		if tag == 'replace':
			leven_cost += np.maximum(i2-i1, j2-j1)
		elif tag == 'insert':
			leven_cost += (j2-j1)
		elif tag == 'delete':
			leven_cost += (i2-i1)
	return leven_cost

if __name__=='__main__':
    data_input = compute_time_freq_images(filepath=filepath)
    
    plt.imshow(data_input.T, origin = 'lower')
    plt.savefig('test.png')