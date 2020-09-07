import numpy as np

import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell,GraphKernel
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
# save graph files.
context.set_context(save_graphs=True)
# enable graph kernel fusion.
context.set_context(enable_graph_kernel=True)


# example for basic fusion.
class NetBasicFuse(Cell):
    def __init__(self):
        super(NetBasicFuse, self).__init__()
        self.add = P.TensorAdd()
        self.mul = P.Mul()
    def construct(self, x):
        mul_res = self.mul(x, 2.0)
        add_res = self.add(mul_res, 1.0)
        out_basic = self.mul(add_res, 3.0)
        return out_basic


# example for composite fusion.
class Composite(GraphKernel):
    def __init__(self):
        super(Composite, self).__init__()
        self.add = P.TensorAdd()
        self.mul = P.Mul()

    def construct(self, x):
        mul_res = self.mul(x, 2.0)
        add_res = self.add(mul_res, 1.0)
        return add_res

class NetCompositeFuse(Cell):
    def __init__(self):
        super(NetCompositeFuse, self).__init__()
        self.mul = P.Mul()
        self.composite=Composite()
    def construct(self, x):
        composite_=self.composite(x)
        pow_res = self.mul(composite_, 3.0)
        return pow_res


def test_basic_fuse():
    x = np.random.randn(4, 4).astype(np.float32)
    net = NetBasicFuse()
    result = net(Tensor(x))
    print("================result=======================")
    print("x: {}".format(x))
    print("result: {}".format(result))
    print("=======================================")


def test_composite_fuse():
    x = np.random.randn(4, 4).astype(np.float32)
    net = NetCompositeFuse()
    result = net(Tensor(x))
    print("================result=======================")
    print("x: {}".format(x))
    print("result: {}".format(result))
    print("=======================================")