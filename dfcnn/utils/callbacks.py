"""
callbacks
"""
import os
from mindspore import Model
from mindspore.train.callback import Callback, ModelCheckpoint
import mindspore.common.dtype as mstype
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint, Tensor
from .log import logger
from .utils import CTCLabelConverter, get_edit_distance
from .data import get_dataset
from .const import MODEL_PATH, BEST_MODEL_PATH

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
if not os.path.exists(BEST_MODEL_PATH):
    os.mkdir(BEST_MODEL_PATH)


class Logging(Callback):
    def __init__(self, logger: logger, model_ckpt: ModelCheckpoint):
        super(Logging, self).__init__()
        self.logger = logger
        self.model_ckpt = model_ckpt
        self.best_acc = None

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num > 0 and step_num % self.model_ckpt._config.save_checkpoint_steps == 0:
            self.logger.logging(latest_ckpt_file_name=self.model_ckpt.latest_ckpt_file_name)


class StepAccInfo(Callback):
    def __init__(self, model: Model, name, div,
                 test_dev_batch_size, step_eval,
                 eval_step, eval_epoch,
                 logger: logger, patience, dataset_size, threshold=0.8):
        self.model = model
        self.div = div
        self.test_dev_batch_size = test_dev_batch_size
        self.step_eval = step_eval
        self.eval_step = eval_step
        self.eval_epoch = eval_epoch

        self.logger = logger

        self.dataset_size = dataset_size

        self.best_acc = -99999
        self.threshold = threshold
        self.best_ckpt = os.path.join(BEST_MODEL_PATH, f'{name}.ckpt')
        self.patience = patience
        self.patience_count = 0

    def step_end(self, run_context):
        if not self.step_eval:
            return None
        flag = ''
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch - 1) * self.dataset_size + cb_params.cur_step_num
        if cur_step > 0 and cur_step % self.eval_step == 0:
            acc = self.eval()

            if acc > self.best_acc:
                flag = '↑'
                self.best_acc = acc
                save_checkpoint(self.model._network, self.best_ckpt)
                self.logger.update_acc_ckpt(acc, self.best_ckpt)

            else:
                if acc > self.threshold:
                    self.patience_count += 1
                    if self.patience_count > self.patience:
                        param_dict = load_checkpoint(ckpt_file_name=self.best_ckpt, net=self.model._network)
                        load_param_into_net(net=self.model._network, parameter_dict=param_dict)
                        self.patience_count = 0

            print(f'* acc for epoch: {cur_epoch} is {acc * 100}%{flag}, best acc is {self.best_acc * 100}%')

    def epoch_end(self, run_context):
        if self.step_eval:
            return None
        flag = ''
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch > 0 and cur_epoch % self.eval_epoch == 0:
            acc = self.eval()

            if acc > self.best_acc:
                flag = '↑'
                self.best_acc = acc
                save_checkpoint(self.model._network, self.best_ckpt)
                self.logger.update_acc_ckpt(acc, self.best_ckpt)

            else:
                if acc > self.threshold:
                    self.patience_count += 1
                    if self.patience_count > self.patience:
                        param_dict = load_checkpoint(ckpt_file_name=self.best_ckpt, net=self.model._network)
                        load_param_into_net(net=self.model._network, parameter_dict=param_dict)
                        self.patience_count = 0

            print(f'* acc for epoch: {cur_epoch} is {acc * 100}%{flag}, best acc is {self.best_acc * 100}%')

    def eval(self):
        print('* start evaluating...')
        self.model._network.set_train(False)

        eval_dataset, idx2label, label2idx = get_dataset(test_dev_batch_size=self.test_dev_batch_size, phase='dev',
                                                         div=self.div, num_parallel_workers=4)

        converter = CTCLabelConverter(label2idx=label2idx, idx2label=idx2label, batch_size=self.test_dev_batch_size)
        words_num = 0
        word_error_num = 0
        for data in eval_dataset.create_tuple_iterator():
            img_batch, label_indices, label_batch, sequence_length, lab_len = data
            img_tensor = Tensor(img_batch, mstype.float32)
            model_predict = self.model._network.predict(img_tensor)

            pred_str = converter.ctc_decoder(model_predict)
            label_str = converter.decode_label(label_batch, lab_len)

            for pred, lab in zip(pred_str, label_str):
                words_n = len(lab)
                words_num += words_n

                edit_distance = get_edit_distance(lab, pred)

                if edit_distance <= words_n:
                    word_error_num += edit_distance
                else:
                    word_error_num += words_n

        self.model._network.set_train(True)

        return 1 - word_error_num / words_num
