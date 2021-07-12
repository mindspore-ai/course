"""
文件名：CreateDataset.py
作者：孙策
描述：用于创建训练或测试时所用的数据集 的函数
修改人：〈修改人〉
修改时间：YYYY-MM-DD
修改内容：〈修改内容〉
"""
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as CV2
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
from DatasetGenerator import DatasetGenerator

def create_dataset(dataset_path, csv_path, batch_size, repeat_size, device_num, rank_id):
    """ 该函数用于创建训练或测试时所用的数据集
    Args:
        data_path: 数据集所在的文件夹
        csv_path: 描述数据集的txt文件。训练集还是测试机就是根据该txt文件进行区分的。
        batch_size: 训练时的batch_size参数
        repeat_size: 数据的重复次数
        num_parallel_workers: 并行工作数量
    """
    
    # 创建数据集生成器。
    dataset_generator = DatasetGenerator(dataset_path, csv_path)
    
    # 将创建的数据集生成器传入到GeneratorDataset类中，创建mindspore数据集。
    # ["image", "label"]表示数据集中的数据名称用image标识，标签数据用label标识。
    dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], num_shards=device_num, shard_id=rank_id, shuffle=True)

    # 确定对图像数据进行变换的一些参数
    resize_height, resize_width = 224, 224  # 图片尺寸
    rescale = 1.0 / 255.0  # 归一化缩放系数
    shift = 0.0  # 偏移量

    # 定义调整图片的尺寸大小的操作
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Resize images to (32, 32)
    # 根据rescale和shift，定义对图像中的像素值进行归一化处理的操作
    rescale_op = CV.Rescale(rescale, shift)
    # 定义对图像中的像素进行标准化处理的操作。下面传入的6个参数，是该数据集所有图片的RGB三个通道的均值和方差。
    normalize_op = CV.Normalize((0.46, 0.46, 0.46), (0.27, 0.27, 0.27))
    # 为了适应网络，将数据由于(height, width, channel) 变换为(channel, height, width)
    changeswap_op = CV.HWC2CHW()
    # 将label的数据类型改成int32类型
    type_cast_op = CV2.TypeCast(mstype.int32)
    
    c_trans = []
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]
    
    # 将上述定义的操作应用到数据集上，对数据集中的图像进行一定的变换。
    # 标签数据类型的变换
    dataset = dataset.map(operations=type_cast_op, input_columns="label")
    # 图片的变换
    dataset = dataset.map(operations=c_trans, input_columns="image")


    # 数据集相关参数设置 
    dataset = dataset.shuffle(buffer_size=10)    # 设置缓存大小
    dataset = dataset.batch(batch_size, drop_remainder=True)  # 设置batch_size
    dataset = dataset.repeat(repeat_size)  # 设置repeat_size

    return dataset