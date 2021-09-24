"""
Training cell for the DFCNN+CTC
"""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Parameter, ParameterTuple, Tensor, context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


GRADIENT_CLIP_MIN = -64000
GRADIENT_CLIP_MAX = 64000


class ClipGradients(nn.Cell):
    """
    Clip large gradients, typically generated from overflow.
    """

    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self, grads, clip_min, clip_max):
        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)

            t = C.clip_by_value(grad, self.cast(F.tuple_to_array((clip_min,)), dt),
                                self.cast(F.tuple_to_array((clip_max,)), dt))
            t = self.cast(t, dt)
            new_grads = new_grads + (t,)
        return new_grads


class DFCNNCTCTrainOneStepWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of DFCNN-CTC network training.
    Used for GPU training in order to manage overflowing gradients.
    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_sense (Cell): Loss scaling value.
    """

    def __init__(self, network, optimizer, scale_sense):
        super(DFCNNCTCTrainOneStepWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.optimizer = optimizer

        if isinstance(scale_sense, nn.Cell):
            self.loss_scaling_manager = scale_sense
            self.scale_sense = Parameter(Tensor(scale_sense.get_loss_scale(),
                                                dtype=mstype.float32), name="scale_sense")
        elif isinstance(scale_sense, Tensor):
            if scale_sense.shape == (1,) or scale_sense.shape == ():
                self.scale_sense = Parameter(scale_sense, name='scale_sense')
            else:
                raise ValueError("The shape of scale_sense must be (1,) or (), but got {}".format(
                    scale_sense.shape))
        else:
            raise TypeError("The scale_sense must be Cell or Tensor, but got {}".format(
                type(scale_sense)))

        self.network.set_grad()
        self.weights = ParameterTuple(network.trainable_params())

        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)

        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode not in ParallelMode.MODE_LIST:
            raise ValueError("Parallel mode does not support: ", self.parallel_mode)
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)

        self.clip_gradients = ClipGradients()
        self.cast = P.Cast()
        self.addn = P.AddN()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()
        self.less_equal = P.LessEqual()
        self.allreduce = P.AllReduce()

    def predict(self, img):
        return self.network.predict(img)

    def construct(self, img, label_indices, text, sequence_length, lab_len):
        """
        Cell's forward
        Args:
            img: input
            label_indices: get from the data generator
            text: label got from the data generator
            sequence_length: get from the data generator
            lab_len: get from the data generator

        Returns:
            loss: loss value
        """
        weights = self.weights
        loss = self.network(img, label_indices, text, sequence_length)

        scaling_sens = self.scale_sense

        grads = self.grad(self.network, weights)(img, label_indices, text, sequence_length,
                                                 self.cast(scaling_sens, mstype.float32))

        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.clip_gradients(grads, GRADIENT_CLIP_MIN, GRADIENT_CLIP_MAX)

        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

        success = self.optimizer(grads)

        ret = (loss, scaling_sens)
        return F.depend(ret, success)


class WithLossCell(nn.Cell):

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img, label_indices, text, sequence_length):
        model_predict = self._backbone(img)
        return self._loss_fn(model_predict, label_indices, text, sequence_length)

    @property
    def backbone_network(self):
        return self._backbone

    def predict(self, img):
        return self._backbone(img)
