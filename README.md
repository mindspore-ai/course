# course

基于MindSpore开源深度学习框架的实验指导，仅用于教学或培训目的。

部分内容来源于开源社区、网络或第三方。如果有内容侵犯了您的权力，请通过issue留言，或者提交pull request。

请前往[MindSpore开源社区](https://www.mindspore.cn/)获取更多视频和文档教程。

## 内容

建议先学习[MindSpore入门](mindspore)了解MindSpore及其用法。再学习[手写数字识别](lenet5)和[模型保存和加载](checkpoint)，了解如何通过ModelArts训练作业、ModelArts Notebook、或本地环境进行实验，以及三者的注意事项。

对于MindSpore端侧，建议先体验[端侧图像分类应用部署](lite_demo_deploy)，再了解[端侧C++推理流程](lite_cpp_inference)并完成课后任务，完成由浅入深的端侧推理部分学习过程。

### 深度学习

1. [手写数字识别[LeNet5][Ascend/CPU/GPU]](lenet5)
2. [模型保存和加载[LeNet5][Ascend/CPU/GPU]](checkpoint)
3. [优化器对比[Dense]](optimizer)
4. [正则化对比[Conv1x1]](regularization)

### 计算机视觉

1. [FashionMNIST图像分类[MLP]](feedforward)
2. [CIFAR-10图像分类[ResNet50]](resnet50)
3. [花卉分类[MobileNetV2]](fine_tune)
4. [语义分割[DeepLabV3]](deeplabv3)
5. [端侧图像分类应用部署[MobileNetV2][Lite]](lite_demo_deploy)
6. [端侧目标检测[ssd_MobileNetV2][Lite]](lite_cpp_inference)

### 自然语言处理

1. [情感分类[LSTM][CPU/GPU]](lstm)
2. [中英翻译[Transformer]](transformer)
3. [新闻分类、命名实体识别[BERT, CRF]](bert)

### 图神经网络

1. [科学出版物分类[GCN]](graph_convolutional_network)
2. [科学出版物分类[GAT]](graph_attention_network)

### 机器学习

1. [线性方程拟合[Linear Regression]](linear_regression)
2. [鸢尾花二分类[Logistic Regression]](logistic_regression)
3. [鸢尾花三分类[Softmax Regression]](softmax_regression)
4. [葡萄酒分类[KNN]](knn)

### 性能加速

1. [图算融合](graph_kernel)
2. [混合精度](mixed_precision)

### 大作业

1. [模型和训练策略调优[CNN]](tuning)

## 版权

- [Apache License 2.0](LICENSE)
- [Creative Commons License version 4.0](LICENSE-CC-BY-4.0)
