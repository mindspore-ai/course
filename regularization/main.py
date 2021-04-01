import random
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms

from mindspore import nn
from mindspore import context
from mindspore.common.initializer import Normal
from IPython.display import clear_output
# %matplotlib inline

# set execution mode and device platform.
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
# ```

# 设置超参和全局变量：

# ```python
N_SAMPLES = 40 # number of samples
BATCH_SIZE = 40
NOISE_RATE = 0.2
INPUT_DIM = 1
HIDDEN_DIM = 100 # number of units in one hidden layer
OUTPUT_DIM = 1
N_LAYERS = 6 # number of hidden layers
ITERATION = 1500
LEARNING_RATE = 0.003
DROPOUT_RATE = 0.7
WEIGHT_DECAY = 1e-4  # coefficient of L2 penalty
MAX_COUNT = 20  # max count that loss does not decrease. Used for early stop.
ACTIVATION = nn.LeakyReLU # activation function

#固定结果
def fix_seed(seed=1):
    # reproducible
    random.seed(seed)
    np.random.seed(seed)

# 小批量样本索引
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

fix_seed(5)
data_x = np.linspace(-1, 1, num=int(N_SAMPLES*2.5))[:, np.newaxis]
data_y = np.cos(np.pi*data_x)
p = np.random.permutation(len(data_x))

train_x, train_y = data_x[p[0:N_SAMPLES]], data_y[p[0:N_SAMPLES]]
test_x, test_y = data_x[p[N_SAMPLES:N_SAMPLES*2]], data_y[p[N_SAMPLES:N_SAMPLES*2]]
validate_x, validate_y = data_x[p[N_SAMPLES*2:]], data_y[p[N_SAMPLES*2:]]

noise = np.random.normal(0, NOISE_RATE, train_y.shape)
train_y += noise


class CosineNet(nn.Cell):
    def __init__(self, batchnorm, dropout):
        super(CosineNet, self).__init__()
        layers = []
        if batchnorm:
            layers.append(nn.BatchNorm2d(INPUT_DIM))
        
        # initialize hidden layers
        for l_n in range(N_LAYERS):
            in_channels = HIDDEN_DIM if l_n > 0 else INPUT_DIM
            # Use 1x1Conv instead of Dense, which coordinate better with BatchNorm2d opetator;
            conv = nn.Conv2d(in_channels, HIDDEN_DIM, kernel_size=1, pad_mode='valid', has_bias=True, weight_init=Normal(0.01))
            layers.append(conv)
            if batchnorm:
                layers.append(nn.BatchNorm2d(HIDDEN_DIM))
            if dropout:
                layers.append(nn.Dropout(DROPOUT_RATE))
            layers.append(ACTIVATION())
        self.layers = nn.SequentialCell(layers)
        
        # initialize output layers
        self.flatten = nn.Flatten() # convert 4-dim tensor (N,C,H,W) to 2-dim tensor(N,C*H*W)
        self.fc = nn.Dense(HIDDEN_DIM, OUTPUT_DIM, weight_init=Normal(0.1), bias_init='zeros')
        
    def construct(self, x):
        # construct hidden layers
        x = self.layers(x)
        # construct output layers
        x = self.flatten(x)
        x = self.fc(x)
        return x

def build_fn(batchnorm, dropout, l2):
    # initilize the net, Loss, optimizer
    net = CosineNet(batchnorm=batchnorm, dropout=dropout)
    loss = nn.loss.MSELoss()
    opt = nn.optim.Adam(net.trainable_params(), learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY if l2 else 0.0)
    # Build a net with loss and optimizer.
    with_loss = nn.WithLossCell(net, loss)
    train_step = nn.TrainOneStepCell(with_loss, opt).set_train()
    return train_step, with_loss, net

# Build 5 different training jobs.
fc_train, fc_loss, fc_predict = build_fn(batchnorm=False, dropout=False, l2=False) # 默认任务
dropout_train, dropout_loss, dropout_predict = build_fn(batchnorm=False, dropout=True, l2=False) # 实验dropout功能
bn_train, bn_loss, bn_predict = build_fn(batchnorm=True, dropout=False, l2=False) # 实验batchnorm功能
l2_train, l2_loss, l2_predict = build_fn(batchnorm=False, dropout=False, l2=True) # 实验l2 regularization功能
early_stop_train, early_stop_loss, early_stop_predict = build_fn(batchnorm=False, dropout=False, l2=False) # 实验Early Stop功能
# Used for batchnorm, dropout and other operators to determine whether the net is in the train state or not.
nets_train = [fc_train, dropout_train, bn_train, l2_train, early_stop_train]
nets_loss = [fc_loss, dropout_loss, bn_loss, l2_loss, early_stop_loss]
nets_predict = [fc_predict, dropout_predict, bn_predict, l2_predict, early_stop_predict]

def set_train(nets, mode=True):
    for net in nets:
        net.set_train(mode=mode)

# ```python
# Convert the tensor shape from (N, C) to (N, C, H, W).
data_xt, data_yt = ms.Tensor(data_x.reshape(data_x.shape + (1, 1)), ms.float32), ms.Tensor(data_y, ms.float32)
test_xt, test_yt = ms.Tensor(test_x.reshape(test_x.shape + (1, 1)), ms.float32), ms.Tensor(test_y, ms.float32)
validate_xt, validate_yt = ms.Tensor(validate_x.reshape(validate_x.shape + (1, 1)), ms.float32), ms.Tensor(validate_y, ms.float32)
# ```

# 启动5个训练任务，并通过plot观察各模型拟合效果。因为同时启动5个任务，每个任务又有训练、计算loss、预测几个子任务，刚开始模型编译的时间较久。

# ```python
# Define parameters of Early Stop.
# If it is True, stop the training process.
early_stop = False
# Used to record min validation loss during training. Should be initialized with Big enough value.
min_val_loss = 1
# In the training iteration process, how many consecutive times are the validation loss greater than min_val_loss.
count = 0
for it in range(ITERATION):
    # In each iteration randomly selects a batch sample from the training set.
    mb_idx = sample_idx(N_SAMPLES, BATCH_SIZE)
    x_batch, y_batch = train_x[mb_idx, :], train_y[mb_idx, :]
    x_batch, y_batch = ms.Tensor(x_batch.reshape(x_batch.shape + (1, 1)), ms.float32), ms.Tensor(y_batch, ms.float32)
    
    # Set the nets to training state.
    set_train(nets_train, True)
    fc_train(x_batch, y_batch)
    dropout_train(x_batch, y_batch)
    bn_train(x_batch, y_batch)
    l2_train(x_batch, y_batch)
    # Skip the training for Early Stop model when early_step==True.
    if not early_stop:
        early_stop_train(x_batch, y_batch)
    
    if it % 20 == 0:
        # Set the nets to none training state.
        set_train(nets_loss+nets_predict, False)
        # Compute the test loss of all models.
        loss_fc = fc_loss(test_xt, test_yt)
        loss_dropout = dropout_loss(test_xt, test_yt)
        loss_bn = bn_loss(test_xt, test_yt)
        loss_l2 = l2_loss(test_xt, test_yt)
        loss_early_stop = early_stop_loss(test_xt, test_yt)
        print("loss_fc", loss_fc)
        print("loss_dropout", loss_dropout)
        print("loss_bn", loss_bn)
        print("loss_l2", loss_l2)
        print("loss_early_stop", loss_early_stop)
        # Compute the predict values on all samples. 
        all_fc = fc_predict(data_xt)
        all_dropout = dropout_predict(data_xt)
        all_bn = bn_predict(data_xt)
        all_l2 = l2_predict(data_xt)
        all_early_stop = early_stop_predict(data_xt)
        
        # For the Early Stop model, when the validation loss is greater than min_val_loss MAX_COUNT consecutive times,
        # stop the training process.
        if not early_stop:
            val_loss = early_stop_loss(validate_xt, validate_yt)
            if val_loss > min_val_loss:
                count += 1
            else:
                min_val_loss = val_loss
                count = 0
            
            if count == MAX_COUNT:
                early_stop = True
                print('='*10, 'early stopped', '='*10)
        
        # Draw the figure.
        plt.figure(1, figsize=(15,10))
        plt.cla()
        plt.scatter(train_x, train_y, c='magenta', s=50, alpha=0.3, label='train samples')
        plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.3, label='test samples')
        plt.plot(data_x, all_fc.asnumpy(), 'r', label='overfitting')
        plt.plot(data_x, all_l2.asnumpy(), 'y', label='L2 regularization')
        plt.plot(data_x, all_early_stop.asnumpy(), 'k', label='early stopping')
        plt.plot(data_x, all_dropout.asnumpy(), 'b', label='dropout({})'.format(DROPOUT_RATE))
        plt.plot(data_x, all_bn.asnumpy(), 'g', label='batch normalization')
        plt.text(-0.1, -1.2, 'overfitting loss=%.4f' % loss_fc.asnumpy(), fontdict={'size': 20, 'color': 'red'})
        plt.text(-0.1, -1.5, 'L2 regularization loss=%.4f' % loss_l2.asnumpy(), fontdict={'size': 20, 'color': 'y'})
        plt.text(-0.1, -1.8, 'early stopping loss=%.4f' % loss_early_stop.asnumpy(), fontdict={'size': 20, 'color': 'black'})
        plt.text(-0.1, -2.1, 'dropout loss=%.4f' % loss_dropout.asnumpy(), fontdict={'size': 20, 'color': 'blue'})
        plt.text(-0.1, -2.4, 'batch normalization loss=%.4f' % loss_bn.asnumpy(), fontdict={'size': 20, 'color': 'green'})

        plt.legend(loc='upper left');
        plt.ylim((-2.5, 2.5));
        clear_output(wait=True)
        plt.show()
        plt.savefig('a.png', format='png')
