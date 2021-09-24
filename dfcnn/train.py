"""
Train
"""
import os
import time

import mindspore.common.dtype as mstype
import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, nn
from mindspore.common import set_seed
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint, SummaryCollector
from mindspore.train.model import Model

from model import DFCNN, CTCLoss
from model.training_cell import DFCNNCTCTrainOneStepWithLossScaleCell, WithLossCell
from utils.const import SUMMARY_DIR
from utils.callbacks import Logging, StepAccInfo
from utils.config import config, get_eval_config
from utils.data import get_dataset
from utils.log import logger
from utils.lr_schedule import dynamic_lr


def main():
    set_seed(1)
    date = time.strftime("%Y%m%d%H%M%S", time.localtime())
    print(f'* Preparing to train model {date}')

    # ************** configuration ****************
    # - training setting
    resume = config['resume']
    if config['mode'] == 'PYNATIVE':
        mode = context.PYNATIVE_MODE
    else:
        mode = context.GRAPH_MODE

    device = config['device']
    device_id = config['device_id']
    dataset_sink_mode = config['dataset_sink_mode']

    # use in dataset
    div = 8

    # setting bias and padding
    if resume:
        print('* Resuming model...')
        resume_config_log = config['resume_config_log']
        resume_config = get_eval_config(resume_config_log)
        if 'best_ckpt' in resume_config.keys():
            resume_model_path = resume_config['best_ckpt']
        else:
            resume_model_path = resume_config['latest_model']
            print('* [WARNING] Not using the best model, but latest saved model instead.')

        has_bias = resume_config['has_bias']
        use_dropout = resume_config['use_dropout']

        pad_mode = resume_config['pad_mode']

        if pad_mode == 'pad':
            padding = resume_config['padding']
        elif pad_mode == 'same':
            padding = 0
        else:
            raise ValueError(f"invalid pad mode: {pad_mode}!")

        best_acc = resume_config['best_acc']
        best_ckpt = resume_config['best_ckpt']
        print('* The best accuracy in dev dataset for the current resumed model is {:.2f}%'.format(best_acc * 100))

    else:
        has_bias = config['has_bias']
        use_dropout = config['use_dropout']

        pad_mode = config['pad_mode']

        if pad_mode == 'pad':
            padding = config['padding']
        elif pad_mode == 'same':
            padding = 0
        else:
            raise ValueError(f"invalid pad mode: {pad_mode}!")

    # hyper-parameters
    if resume:
        batch_size = resume_config['batch_size']
        opt_type = resume_config['opt']
        use_dynamic_lr = resume_config['use_dynamic_lr']
        warmup_step = resume_config['warmup_step']
        warmup_ratio = resume_config['warmup_ratio']
    else:
        batch_size = config['batch_size']
        opt_type = config['opt']
        use_dynamic_lr = config['use_dynamic_lr']
        warmup_step = config['warmup_step']
        warmup_ratio = config['warmup_ratio']

    test_dev_batch_size = config['test_dev_batch_size']
    learning_rate = float(config['learning_rate'])
    epochs = config['epochs']
    loss_scale = config['loss_scale']

    # configuration of saving model checkpoint
    save_checkpoint_steps = config['save_checkpoint_steps']
    keep_checkpoint_max = config['keep_checkpoint_max']
    prefix = config['prefix'] + '_' + date
    model_dir = config['model_dir']

    # loss monitor
    loss_monitor_step = config['loss_monitor_step']

    # whether to use mindInsight summary
    use_summary = config['use_summary']

    # step_eval
    use_step_eval = config['use_step_eval']
    eval_step = config['eval_step']
    eval_epoch = config['eval_epoch']
    patience = config['patience']

    # eval in steps or epochs
    step_eval = True

    if eval_step == -1:
        step_eval = False

    # ************** end of configuration **************
    if device == 'GPU':
        context.set_context(mode=mode, device_target=device, device_id=device_id)
    elif device == 'Ascend':
        import moxing as mox
        from utils.const import DATA_PATH, MODEL_PATH, BEST_MODEL_PATH, LOG_PATH
        obs_datapath = config['obs_datapath']
        obs_saved_model = config['obs_saved_model']
        obs_best_model = config['obs_best_model']
        obs_log = config['obs_log']
        mox.file.copy_parallel(obs_datapath, DATA_PATH)
        mox.file.copy_parallel(MODEL_PATH, obs_saved_model)
        mox.file.copy_parallel(BEST_MODEL_PATH, obs_best_model)
        mox.file.copy_parallel(LOG_PATH, obs_log)
        context.set_context(mode=mode, device_target=device)
        use_summary = False

    # callbacks function
    callbacks = []

    # data
    train_loader, idx2label, label2idx = get_dataset(batch_size=batch_size, phase='train',
                                                     test_dev_batch_size=test_dev_batch_size, div=div,
                                                     num_parallel_workers=4)

    if eval_step == 0:
        eval_step = train_loader.get_dataset_size()

    # network
    net = DFCNN(num_classes=len(label2idx), padding=padding, pad_mode=pad_mode,
                has_bias=has_bias, use_dropout=use_dropout)

    # Criterion
    criterion = CTCLoss()

    # resume
    if resume:
        print("* Loading parameters...")
        param_dict = load_checkpoint(resume_model_path)
        # load the parameter into net
        load_param_into_net(net, param_dict)
        print(f'* Parameters loading from {resume_model_path} succeeded!')

    net.set_train(True)
    net.set_grad(True)

    # lr schedule
    if use_dynamic_lr:
        dataset_size = train_loader.get_dataset_size()
        learning_rate = Tensor(dynamic_lr(base_lr=learning_rate, warmup_step=warmup_step,
                                          warmup_ratio=warmup_ratio, epochs=epochs,
                                          steps_per_epoch=dataset_size), mstype.float32)
        print('* Using dynamic learning rate, which will be set up as :', learning_rate.asnumpy())

    # optim
    if opt_type == 'adam':
        opt = nn.Adam(net.trainable_params(), learning_rate=learning_rate, beta1=0.9, beta2=0.999, weight_decay=0.0,
                      eps=10e-8)
    elif opt_type == 'rms':
        opt = nn.RMSProp(params=net.trainable_params(),
                         centered=True,
                         learning_rate=learning_rate,
                         momentum=0.9,
                         loss_scale=loss_scale)
    elif opt_type == 'sgd':
        opt = nn.SGD(params=net.trainable_params(), learning_rate=learning_rate)
    else:
        raise ValueError(f"optimizer: {opt_type} is not supported for now!")

    if resume:
        # load the parameter into optimizer
        load_param_into_net(opt, param_dict)

    # save_model
    config_ck = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps, keep_checkpoint_max=keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix=prefix, directory=model_dir, config=config_ck)

    # logger
    the_logger = logger(config, date)
    log = Logging(logger=the_logger, model_ckpt=ckpt_cb)

    callbacks.append(ckpt_cb)
    callbacks.append(log)

    net = WithLossCell(net, criterion)
    scaling_sens = Tensor(np.full((1), loss_scale), dtype=mstype.float32)

    net = DFCNNCTCTrainOneStepWithLossScaleCell(net, opt, scaling_sens)
    net.set_train(True)
    model = Model(net)

    if use_step_eval:
        # step evaluation
        step_eval = StepAccInfo(model=model, name=prefix, div=div, test_dev_batch_size=test_dev_batch_size,
                                step_eval=step_eval, eval_step=eval_step, eval_epoch=eval_epoch,
                                logger=the_logger, patience=patience, dataset_size=train_loader.get_dataset_size())

        callbacks.append(step_eval)

    # loss monitor
    loss_monitor = LossMonitor(loss_monitor_step)

    callbacks.append(loss_monitor)

    if use_summary:
        summary_dir = os.path.join(SUMMARY_DIR, date)
        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)
        # mindInsight
        summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=1, max_file_size=4 * 1024 ** 3)
        callbacks.append(summary_collector)

    if resume:
        the_logger.update_acc_ckpt(best_acc, best_ckpt)

    print(f'* Start training...')
    model.train(epochs,
                train_loader,
                callbacks=callbacks,
                dataset_sink_mode=dataset_sink_mode)


if __name__ == '__main__':
    main()
