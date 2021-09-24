"""
data generator
"""
import os
import mindspore.context as context
import numpy as np
from utils import compute_time_freq_images, divmod
from .const import DATA_PATH
import mindspore.dataset as ds


class Thchs30(object):
    def __init__(self, dataset_path, batch_size, div=8, phase='train', wav_max_len=1600, lab_max_len=50):
        """Initialization

          Args:
              dataset_path (str): the root path of the dataset
              div (int, optional): the divisor to use in the DFCNN model which depends on the structure of the model,
                                   leads to the max_time dimension is reduced by a factor of 8 after forward
                                   propagation. Defaults to 8.
              phase (str, optional): Phase to which the dataset should be used: {train,test,dev}. Defaults to 'train'.
              wav_max_len (int, optional): Maximum length of the language spectrum map. Defaults to 1640.
              lab_max_len (int, optional): Maximum length of label array. Defaults to 50.
          """
        self.dataset_path = dataset_path
        self.label_list_path = os.path.join(dataset_path, 'label.txt')
        self.phase = phase  # {train,test,dev}
        self.wav_max_len = wav_max_len
        self.lab_max_len = lab_max_len
        self.div = div
        self.batch_size = batch_size

        # Get a list of audio files, corresponding markup files
        self.wav_list, self.label_lst = self.get_thchs_data(dir=self.phase)

        self.dataset_size = len(self.wav_list) // self.batch_size

        # Generate label data
        self.label_data = self.generate_thchs_label_data(label_lst=self.label_lst, dir=self.phase)

        # Create an index of label ids for the entire dataset
        if not os.path.exists(self.label_list_path):
            # Generate label list file
            self.generate_label_file()

        self.idx2label, self.label2idx = self.build_vocab()

    def get_vocab(self):
        return self.idx2label, self.label2idx

    def generate_label_file(self):
        _, all_label_lst = self.get_thchs_data(dir='data')
        all_label_data = self.generate_thchs_label_data(label_lst=all_label_lst, dir='data')
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

            # Blank string
            f.write('_')

    def get_thchs_data(self, dir):
        """
        Get a list of audio files and annotation files from the thchs30 data source

        Args:
            dir (str): Data set path: {'train','dev','test'}

        Returns:
            tuple: (labeled list, audio list)
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

    def check(self, label_lst, wav_lst):
        """
        Check that the audio and label correspond to the same

        Args:
            label_lst (list): Label List
            wav_lst (list): Audio file list
        """
        label_len = len(label_lst)
        wav_len = len(wav_lst)

        assert label_len == wav_len, "Audio files do not match the number of audio labels, " \
                                     "please check if the data set is complete."

        for i in range(label_len):
            wav_name = (wav_lst[i].split('/')[-1]).split('.')[0]
            label_name = (label_lst[i].split('/')[-1]).split('.')[0]
            assert wav_name == label_name, f"The audio file: {wav_name} does not match the " \
                                           f"label file: {label_name}, please check!"

    def read_thchs_label(self, label_file, dir_name):
        """Read the label file of thchs30

        Args:
            label_file (str): Label file path
            dir_name: dir to read

        Returns:
            str: Label line
        """
        with open(label_file, 'r', encoding='utf8') as f:
            data = f.readlines()
            if dir_name in ['train', 'test', 'dev']:
                # Get the real path
                root = label_file.split(self.phase)[0]
                real_path = os.path.join(root, self.phase, data[0].strip())
                with open(real_path, 'r', encoding='utf8') as f2:
                    data2 = f2.readlines()
                    return data2[1]
            else:
                return data[1]

    def generate_thchs_label_data(self, label_lst, dir):
        """
        Generate a list of label

        Args:
            label_lst (list): List of label files

        Returns:
            list: List of label strings, shaped like:
                ['a2 a4 ....'],
                ['b2 b4 ....'],
        """
        label_data = []
        for label_file in label_lst:
            pny = self.read_thchs_label(label_file=label_file, dir_name=dir)
            label_data.append(pny.strip('\n'))
        return label_data

    def build_vocab(self):
        """
        Create index, id to tag mapping, tag to id mapping.

        Returns:
            tuple: 
                idx2label (dict): The key is id and the value is label.
                label2idx (dict): The key is label and the value is id.
        """
        label_list = self.get_label_list()
        idx2label = {}
        label2idx = {}

        for idx, label in enumerate(label_list):
            idx2label[idx] = label
            label2idx[label] = idx

        return idx2label, label2idx

    def get_label_list(self):
        label_list = []
        # Open label list.
        with open(self.label_list_path, 'r') as f:
            for line in f.readlines():
                label = line.strip()
                label_list.append(label)

            return label_list

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        """
        Random Indexing

        Args:
            index (int): index
        """
        img_batch = []
        label_batch = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            wav_path, label_path = self.wav_list[i], self.label_lst[i]

            # Processing audio.
            # Generating a language map.
            fq_img = compute_time_freq_images(wav_path)
            fq_img = divmod(fq_img, self.div)
            # fq_img = self.wav_pad(fq_img)

            # Processing labels.
            # Label to id.
            label = self.read_thchs_label(label_file=label_path, dir_name=self.phase)
            label, len = self.line2id(label, self.label2idx)

            img_batch.append(fq_img)
            label_batch.append(label)

        img_batch, wav_max_len = self.wav_pad(img_batch)
        label_batch, lab_len = self.encode(label_batch)

        label_indices = []
        for i, _ in enumerate(lab_len):
            for j in range(lab_len[i]):
                label_indices.append((i, j))

        label_indices = np.array(label_indices, dtype=np.int64)
        sequence_length = np.array([wav_max_len // self.div] * self.batch_size, dtype=np.int32)

        return img_batch, label_indices, label_batch, sequence_length, lab_len

    def line2id(self, line, label2idx):
        """
        Mapping labels of single-line speech to ids.

        Args:
            line (list): Single line voice label
            label2idx (dict): Mapping dictionary for label->id

        Returns:
            list: List of mapped ids
        """
        label = [label2idx[pny.strip()] for pny in line.split(' ')]
        return label, len(label)

    def wav_pad(self, wav_data_lst):
        """
        Time-frequency map array padding according to the longest sequence,
        incoming data shape: [batch_size, max_time, 200]
        and the information needed to generate ctc: a list of the lengths of the input sequences
        """
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // self.div for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), 1, self.wav_max_len, 200), dtype=np.float32)
        for i in range(len(wav_data_lst)):
            if len(wav_data_lst[i]) <= self.wav_max_len:
                new_wav_data_lst[i, 0, 0:wav_data_lst[i].shape[0], :] = wav_data_lst[i]
            else:
                new_wav_data_lst[i, 0, 0:wav_data_lst[i].shape[0], :] = wav_data_lst[i][: self.wav_max_len]

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
        """Label arrays are padding by longest sequence
        """
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len), dtype=np.int32)
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens


def get_dataset(dataset_path=DATA_PATH, phase='train', div=8, batch_size=8, test_dev_batch_size=8,
                num_parallel_workers=1):
    """
    Get the training set, the tester, the validation set, and the mapping dictionary of id->label,
    and the mapping dictionary of label->id

    Args:
        dataset_path (str, optional): Dataset path. Defaults to utils.const.DATA_PATH.
        div (int, optional): Divisor to be divided by. Defaults to 1.
        batch_size (int, optional): batch-size. Defaults to 64.
        num_parallel_workers (int, optional): theading-numbers. Defaults to 4.
        phase: Phase to which the dataset should be used: {train,test,dev}. Defaults to 'train'.


    Returns:
        tuple: Training set, testing machine, validation set,
               and mapping dictionary of id->label, mapping dictionary of label->id.
    """
    if phase == 'train':
        train_generator = Thchs30(dataset_path=dataset_path, batch_size=batch_size,
                                  div=div, phase='train')

        train_dataset = ds.GeneratorDataset(source=train_generator,
                                            column_names=['img', 'label_indices', 'label_batch', 'sequence_length',
                                                          'lab_len'],
                                            shuffle=True,
                                            num_parallel_workers=num_parallel_workers)

        # vocab
        idx2label, label2idx = train_generator.get_vocab()

        return train_dataset, idx2label, label2idx

    elif phase == 'test':
        test_generator = Thchs30(dataset_path=dataset_path, batch_size=test_dev_batch_size,
                                 div=div,
                                 phase='test')

        test_dataset = ds.GeneratorDataset(source=test_generator,
                                           column_names=['img', 'label_indices', 'label_batch', 'sequence_length',
                                                         'lab_len'],
                                           shuffle=False,
                                           num_parallel_workers=num_parallel_workers)

        # vocab
        idx2label, label2idx = test_generator.get_vocab()

        return test_dataset, idx2label, label2idx

    elif phase == 'dev':
        dev_generator = Thchs30(dataset_path=dataset_path, batch_size=test_dev_batch_size,
                                div=div,
                                phase='dev')

        dev_dataset = ds.GeneratorDataset(source=dev_generator,
                                          column_names=['img', 'label_indices', 'label_batch', 'sequence_length',
                                                        'lab_len'],
                                          shuffle=True,
                                          num_parallel_workers=num_parallel_workers)

        # vocab
        idx2label, label2idx = dev_generator.get_vocab()

        return dev_dataset, idx2label, label2idx


if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)

    test_dataset, i2l, l2i = get_dataset(num_parallel_workers=1, batch_size=4)
    count = 0
    for test in test_dataset.create_dict_iterator():
        count += 1
        print('{}'.format(test["img"].shape), '{}'.format(test["label_indices"].shape))
        print('{}'.format(test["label_batch"].shape), '{}'.format(test["sequence_length"].shape))

    print(count)
