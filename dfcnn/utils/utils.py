"""
utils
"""
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt

# Audio file path
filepath = '/home/jojo/PythonProjects/speech_data/data_thchs30/data/A11_0.wav'

img_path = '/home/jojo/PythonProjects/DFCNN-CTC/summer/210610338/fq2img'


def compute_time_freq_images(audio_filepath):
    """Generate audio spectrogram

    Args:
        audio_filepath (str): file path

    Returns:
        numpy.array: audio spectrogram
    """
    # -------------- 1. Read audio files ------------
    fs, wav_signal = wav.read(audio_filepath)  # wav_signal is the spectrogram

    # Convert to numpy arrays
    wav_arr = np.array(wav_signal)

    # Spectrum length
    wav_length = wav_arr.shape[0]

    # -------------  2. Hanming Window ----------------
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))

    # Sampling point(s) = fs
    # Sampling point (ms) = fs / 1000
    # Sampling points (frames) = fs / 1000 * Frame Length

    # ------------- 3. Framing of data --------------
    # Frame Length: 25ms
    time_window = 25

    # Frame shift: 10ms
    window_length = fs // 1000 * time_window

    # -------------- 4. Frame addition window ------------------

    # Final number of windows generated
    window_end = int(wav_length / fs * 1000 - time_window) // 10

    # Final output results
    data_input = np.zeros((window_end, window_length // 2), dtype=np.float)

    data_line = np.zeros((1, window_end), dtype=np.float)

    # Split Frame
    for i in range(0, window_end):
        p_start = i * 160
        p_end = p_start + window_length
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # Add Window

        # -------- 5. Fourier transform: generating time-frequency diagrams -------------
        data_line = np.abs(fft(data_line)) / wav_length
        # Set to the value of window_length divided by 2 (i.e. 200) is
        # to take half of the data because it is symmetrical
        data_input[i] = data_line[0:window_length // 2]

    # Take the logarithm and get db
    data_input = np.log(data_input + 1)

    return data_input


def divmod(array, div=8):
    """Integral division processing

    Args:
        array (numpy.array): array
        div (int, optional): divisor. Defaults to 8.

    Returns:
        numpy.array: Integral division of the array
    """
    return array[:array.shape[0] // div * div, :]


class CTCLabelConverter():
    """ Convert between text-label and text-index 
        https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/cnnctc/src/util.py
    """

    def __init__(self, label2idx: dict, idx2label: dict, batch_size):
        """
        ctc decoding tool

        Args:
            label2idx (dict): Mapping of Label->id, the last one should be the empty character
            idx2label (dict): Mapping of id->tags, the last one should be empty character
            batch_size (int): batch size
        """
        self.batch_size = batch_size
        self.label2idx = label2idx
        self.idx2label = idx2label

        self.vocal_list = []

        for key, item in idx2label.items():
            if key == len(idx2label) - 1:
                continue
            self.vocal_list.append(item)

    def encode(self, text):
        """
        convert text-label into text-index.

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

    def decode(self, text_index, length, merge_repeat=True):
        """
        convert text-index into text-label.
        """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if not t[i] == self.label2idx['_']:  # removing repeated characters and blank.
                    if merge_repeat:
                        # 去掉重复项
                        if i > 0 and not t[i - 1] == t[i]:
                            char_list.append(self.idx2label[t[i]])
                    else:
                        char_list.append(self.idx2label[t[i]])

            texts.append(char_list)
            index += l
        return texts

    def decode_label(self, label, label_len):
        label = np.squeeze(label.asnumpy())
        try:
            label_size = np.array([len.asnumpy() for len in label_len])
        except:
            label_size = np.array([len for len in label_len])
        label = np.reshape(label, [-1])
        label_str = self.decode(label, label_size, merge_repeat=False)
        return label_str

    def ctc_decoder(self, model_predicts):
        """
        decode network output

        Args:
            model_predicts (mindspore.Tensor): network output

        Returns:
            list: The list of decoded label strings, with the length of batch_size.
        """
        model_predicts = np.squeeze(model_predicts.asnumpy())

        preds_size = np.array([model_predicts.shape[1]] * self.batch_size)
        preds_index = np.argmax(model_predicts, 2)
        preds_index = np.reshape(preds_index, [-1])
        preds_str = self.decode(preds_index, preds_size)

        return preds_str


import difflib


def get_edit_distance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += np.maximum(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost


if __name__ == '__main__':
    data_input = compute_time_freq_images(audio_filepath=filepath)

    plt.imshow(data_input.T, origin='lower')
    plt.savefig('test.png')
