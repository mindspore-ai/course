"""
Evaluation
"""
import os
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, load_checkpoint, load_param_into_net

from model import DFCNN
from utils import CTCLabelConverter, get_edit_distance
from utils.config import config, get_eval_config
from utils.data import get_dataset


def test():
    if config['mode'] == 'PYNATIVE':
        mode = context.PYNATIVE_MODE
    else:
        mode = context.GRAPH_MODE

    device = config['device']
    device_id = config['device_id']

    if device == 'Ascend':
        import moxing as mox
        from utils.const import DATA_PATH, MODEL_PATH, BEST_MODEL_PATH, LOG_PATH
        obs_datapath = config['obs_datapath']
        obs_saved_model = config['obs_saved_model']
        obs_best_model = config['obs_best_model']
        obs_log = config['obs_log']
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        if not os.path.exists(BEST_MODEL_PATH):
            os.mkdir(BEST_MODEL_PATH)
        if not os.path.exists(LOG_PATH):
            os.mkdir(LOG_PATH)
        mox.file.copy_parallel(obs_datapath, DATA_PATH)
        mox.file.copy_parallel(MODEL_PATH, obs_saved_model)
        mox.file.copy_parallel(BEST_MODEL_PATH, obs_best_model)
        mox.file.copy_parallel(LOG_PATH, obs_log)

    test_dev_batch_size = config['test_dev_batch_size']

    eval_config_log = config['log_to_eval']
    data_num = config['test_dataset_size']

    eval_config = get_eval_config(eval_config_log)

    #    - use in dataset
    div = 8

    if 'best_ckpt' in eval_config.keys():
        eval_model_path = eval_config['best_ckpt']
        if device == 'Ascend':
            import moxing as mox
            from utils.const import BEST_MODEL_PATH
            eval_model_filename = eval_model_path.split('/')[-1]
            obs_best_model = config['obs_best_model']
            mox.file.copy_parallel(obs_best_model + eval_model_filename, eval_model_path)

    else:
        eval_model_path = eval_config['latest_model']
        if device == 'Ascend':
            import moxing as mox
            from utils.const import BEST_MODEL_PATH
            eval_model_filename = eval_model_path.split('/')[-1]
            obs_saved_model = config['obs_saved_model']
            mox.file.copy_parallel(obs_saved_model + eval_model_filename, eval_model_path)
        print('* [WARNING] Not using the best model, but latest saved model instead.')

    #   - 偏差
    has_bias = eval_config['has_bias']
    use_dropout = eval_config['use_dropout']

    #   - pad
    pad_mode = eval_config['pad_mode']

    if pad_mode == 'pad':
        padding = eval_config['padding']
    elif pad_mode == 'same':
        padding = 0
    else:
        raise ValueError(f"invalid pad mode: {pad_mode}!")

    if 'best_acc' in eval_config.keys():
        best_acc = eval_config['best_acc']
        print('* Best accuracy for the dev dataset is : {:.2f}%'.format(best_acc * 100))

    if device == 'GPU':
        context.set_context(mode=mode, device_target=device, device_id=device_id)
    elif device == 'Ascend':
        context.set_context(mode=mode, device_target=device)

    # data
    test_loader, idx2label, label2idx = get_dataset(phase='test', test_dev_batch_size=test_dev_batch_size,
                                                    div=div, num_parallel_workers=4)

    net = DFCNN(num_classes=len(label2idx), padding=padding, pad_mode=pad_mode,
                has_bias=has_bias, use_dropout=use_dropout)

    # loads param
    param_dict = load_checkpoint(eval_model_path)
    load_param_into_net(net, param_dict)
    print('* params loaded!')

    net.set_train(False)

    converter = CTCLabelConverter(label2idx=label2idx, idx2label=idx2label, batch_size=test_dev_batch_size)

    words_num = 0
    word_error_num = 0

    limit = 0
    for data in test_loader.create_tuple_iterator():
        if limit > data_num and not data_num < 0:
            break
        img_batch, label_indices, label_batch, sequence_length, lab_len = data
        img_tensor = Tensor(img_batch, mstype.float32)
        model_predict = net(img_tensor)

        pred_str = converter.ctc_decoder(model_predict)
        label_str = converter.decode_label(label_batch, lab_len)

        for pred, lab in zip(pred_str, label_str):
            if limit > data_num and not data_num < 0:
                break
            words_n = len(lab)
            words_num += words_n

            # get edit distance
            edit_distance = get_edit_distance(lab, pred)

            if edit_distance <= words_n:
                word_error_num += edit_distance
            else:
                word_error_num += words_n
            limit += 1

    if data_num > 0:
        size = str(data_num)
    else:
        size = 'all'
    print('* [Test result] For {} datas, the accuracy is: {:.2f}%'.
          format(size, ((1 - word_error_num / words_num) * 100)))


if __name__ == '__main__':
    test()
