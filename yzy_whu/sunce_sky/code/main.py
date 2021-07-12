import os
import argparse
import time
#import random
from mindspore import context, Model, load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.nn.optim.momentum import Momentum
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from resnet import resnet50
from CreateDataset import create_dataset

# 随机种子初始化
#random.seed(1)

# 定义一个参数接收器。用于读取运行时传入的参数
parser = argparse.ArgumentParser(description='face expression classification')
parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute.')
parser.add_argument('--device_num', type=int, default=24, help='Device num.')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--num_classes', type=int, default=4, help='Num classes.')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='CheckPoint file path.')
#parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path.')
parser.add_argument('--data_url', required=True, help='Location of data.')
parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
parser.add_argument('--mode', type=str, default="train", choices=['train', 'test'], help='train or test')

args = parser.parse_args()
    
dog_dataset_path = "./four_face"    # 定义数据集所在路径

# 设置运行环境的参数
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)


if __name__ == '__main__':
    # 标记开始测试时间
    start_time = time.time()

    # 将数据集中的数据从OBS桶中拷贝到缓存中来。
    import moxing as mox
    mox.file.copy_parallel(src_url=args.data_url, dst_url='./four_face')
    #data_obs_url = 'obs://sunce-demo/expression_recognition/four_face/'
    #mox.file.copy_parallel(data_obs_url, dog_dataset_path)
    
    # 打印所使用的设备
    print(f"use device is : {args.device_target}")
    #dataset_sink_mode = not args.device_target == "CPU"
    
    # 自动并行运算
    if args.run_distribute:
        context.set_auto_parallel_context(device_num=args.device_num, parallel_mode=ParallelMode.DATA_PARALLEL)
        auto_parallel_context().set_all_reduce_fusion_split_indices([140])
        init()

    # 定义训练过程中的一些参数
    epoch_size = args.epoch_size   # 反向传播计算迭代次数
    net = resnet50(args.batch_size, args.num_classes) # 创建ResNet网络对象
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean') # 定义损失行数
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)  # 定义训练过程中的优化器

    # 创建网络模型。metrics={"acc", "loss"}表示评估该网络模型的时候，评估准确率、损失值。
    model = Model(net, loss_fn=net_loss, optimizer=opt, metrics={'acc'})
    #sink_mode = not args.device_target == "CPU"

    # as for train, users could use model.train
    if args.do_train:
        train_dataset = create_dataset(dog_dataset_path, os.path.join(dog_dataset_path, "training.csv"), batch_size=args.batch_size, repeat_size=args.batch_size, device_num=args.device_num, rank_id=device_id)
        batch_num = train_dataset.get_dataset_size()
        config_ck = CheckpointConfig(save_checkpoint_steps=batch_num, keep_checkpoint_max=35)
        ckpoint_cb = ModelCheckpoint(prefix="train_resnet50", directory=args.checkpoint_path, config=config_ck)
        loss_cb = LossMonitor()
        
        print("begin train")
        model.train(epoch_size, train_dataset, callbacks=[ckpoint_cb, loss_cb])
        mox.file.copy_parallel(src_url=args.checkpoint_path, dst_url=args.train_url)
        

    # as for evaluation, users could use model.eval
    if args.do_eval:
        print("Testing Model:")
        if args.checkpoint_path:
            param_dict = load_checkpoint(dog_model_path)
            load_param_into_net(net, param_dict)
        eval_dataset = create_dataset(dog_dataset_path, os.path.join(dog_dataset_path, "validation.csv"), batch_size=args.batch_size, repeat_size=args.batch_size, device_num=args.device_num, rank_id=device_id)
        
        print("begin eval")
        res = model.eval(eval_dataset, dataset_sink_mode=False)  # 测试网络性能，并把结果保存到res_metric
        print("============== Test result:{} ==============".format(res_metric))
        print(f"Total time:{int(time.time() - start_time)}")