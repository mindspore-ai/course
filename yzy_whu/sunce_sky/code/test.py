import os
import mindspore.dataset as ds
from DatasetGenerator import DatasetGenerator
from CreateDataset import create_dataset

if __name__ == '__main__':
    
    dog_dataset_path = "../four_face"
    
    """
    print("begin")
    # 创建数据集生成器
    dataset_genertor = DatasetGenerator(dog_dataset_path, os.path.join(dog_dataset_path, "validation.csv"))
    
    # 将创建的数据集生成器传入到GeneratorDataset类中，创建mindspore数据集。
    # ["image", "label"]表示数据集中的数据名称用image标识，标签数据用label标识。
    dataset = ds.GeneratorDataset(dataset_genertor, ["image", "label"], shuffle=False)
    #print 1 
    print("end")
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        #print("Image shape: {}".format(data['image'].shape), ", Label: {}".format(data['label']))
        print("Image shape: {}".format(data['image']), ", Label: {}".format(data['label']))
    """
    
    print("begin")
    train_dataset = create_dataset(dog_dataset_path, os.path.join(dog_dataset_path, "training.csv"), imageRow=0, labelRow=1)
    eval_dataset = create_dataset(dog_dataset_path, os.path.join(dog_dataset_path, "validation.csv"), imageRow=0, labelRow=1)
    print("end")
    
    #for data in train_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    #    #print("Image shape: {}".format(data['image'].shape), ", Label: {}".format(data['label']))
    #    print("Image shape: {}".format(data['image'].shape), ", Label: {}".format(data['label']))
    #for data in eval_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
    #    #print("Image shape: {}".format(data['image'].shape), ", Label: {}".format(data['label']))
    #    print("Image shape: {}".format(data['image'].shape), ", Label: {}".format(data['label']))