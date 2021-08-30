# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""eval_fgsm"""

import os
import numpy as np
import argparse

from mindspore import Tensor, nn, load_checkpoint, load_param_into_net, Model

from src.net.lenet import LeNet5
from src.data.dataset import create_dataset
from src.utils.fgsm import FastGradientSignMethod

parser = argparse.ArgumentParser("FGSM eps setting")
parser.add_argument("--eps", type=float, default=0.0, help="set the eps of fgsm")
args, _ = parser.parse_known_args()

images = []
labels = []
test_images = []
test_labels = []
predict_labels = []

net = LeNet5()
mnist_path = "./datasets/MNIST_Data/"
param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
load_param_into_net(net, param_dict)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
model = Model(net, net_loss, net_opt, metrics={"Accuracy": nn.Accuracy()})

ds_test = create_dataset(os.path.join(mnist_path, "test"), batch_size=32).create_dict_iterator(output_numpy=True)

for data in ds_test:
    images = data['image'].astype(np.float32)
    labels = data['label']
    test_images.append(images)
    test_labels.append(labels)
    pred_labels = np.argmax(model.predict(Tensor(images)).asnumpy(), axis=1)
    predict_labels.append(pred_labels)

test_images = np.concatenate(test_images)
predict_labels = np.concatenate(predict_labels)
true_labels = np.concatenate(test_labels)

fgsm = FastGradientSignMethod(net, eps=args.eps, loss_fn=net_loss)
advs = fgsm.batch_generate(test_images, true_labels, batch_size=32)

adv_predicts = model.predict(Tensor(advs)).asnumpy()
adv_predicts = np.argmax(adv_predicts, axis=1)
accuracy = np.mean(np.equal(adv_predicts, true_labels))
print(accuracy)
