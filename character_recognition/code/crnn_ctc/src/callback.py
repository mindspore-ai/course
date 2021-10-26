'''
Date: 2021-08-15 19:47:03
LastEditors: xgy
LastEditTime: 2021-08-15 20:09:44
FilePath: \code\crnn_ctc\src\callback.py
'''
"""loss callback"""

import time
from mindspore.train.callback import Callback
from .util import AverageMeter

class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.loss_avg = AverageMeter()
        self.timer = AverageMeter()
        self.start_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()

        loss = cb_params.net_outputs.asnumpy()

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_num = cb_params.cur_step_num

        if cur_step_in_epoch % 2000 == 1:
            self.loss_avg = AverageMeter()
            self.timer = AverageMeter()
            self.start_time = time.time()
        else:
            self.timer.update(time.time() - self.start_time)
            self.start_time = time.time()

        self.loss_avg.update(loss)

        if self._per_print_times != 0 and cur_num % self._per_print_times == 0:
            loss_file = open("./loss.log", "a+")
            loss_file.write("epoch: %s step: %s , loss is %s, average time per step is %s" % (
                cb_params.cur_epoch_num, cur_step_in_epoch,
                self.loss_avg.avg, self.timer.avg))
            loss_file.write("\n")
            loss_file.close()

            print("epoch: %s step: %s , loss is %s, average time per step is %s" % (
                cb_params.cur_epoch_num, cur_step_in_epoch,
                self.loss_avg.avg, self.timer.avg))
