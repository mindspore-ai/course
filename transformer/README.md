# Transformer网络实现英中翻译

## 实验介绍

本实验主要介绍使用MindSpore开发和训练Transformer模型。本实验实现了英中翻译任务。

## 实验目的

- 掌握Transformer模型的基本结构和编程方法。
- 掌握使用Transformer模型进行英中翻译。
- 了解MindSpore的model_zoo模块，以及如何使用model_zoo中的模型。

## 预备知识

- 熟练使用Python，了解Shell及Linux操作系统基本知识。
- 具备一定的深度学习和机器学习理论知识，如BLEU、Embedding、Encoder、Decoder、损失函数、优化器，训练策略、Checkpoint等。
- 了解华为云的基本使用方法，包括[OBS（对象存储）](https://www.huaweicloud.com/product/obs.html)、[ModelArts（AI开发平台）](https://www.huaweicloud.com/product/modelarts.html)、[训练作业](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)等功能。华为云官网：https://www.huaweicloud.com
- 了解并熟悉MindSpore AI计算框架，MindSpore官网：https://www.mindspore.cn/

## 实验环境

- MindSpore 1.0.0（MindSpore版本会定期更新，本指导也会定期刷新，与版本配套）；
- 华为云ModelArts（控制台左上角选择“华北-北京四”）：ModelArts是华为云提供的面向开发者的一站式AI开发平台，集成了昇腾AI处理器资源池，用户可以在该平台下体验MindSpore。

## 实验准备

### 创建OBS桶

本实验需要使用华为云OBS存储脚本和数据集，可以参考[快速通过OBS控制台上传下载文件](https://support.huaweicloud.com/qs-obs/obs_qs_0001.html)了解使用OBS创建桶、上传文件、下载文件的使用方法。

> **提示：** 华为云新用户使用OBS时通常需要创建和配置“访问密钥”，可以在使用OBS时根据提示完成创建和配置。也可以参考[获取访问密钥并完成ModelArts全局配置](https://support.huaweicloud.com/prepare-modelarts/modelarts_08_0002.html)获取并配置访问密钥。

打开[OBS控制台](https://storage.huaweicloud.com/obs/?region=cn-north-4&locale=zh-cn#/obs/manager/buckets)，点击右上角的“创建桶”按钮进入桶配置页面，创建OBS桶的参考配置如下：

- 区域：华北-北京四
- 数据冗余存储策略：单AZ存储
- 桶名称：如ms-course
- 存储类别：标准存储
- 桶策略：公共读
- 归档数据直读：关闭
- 企业项目、标签等配置：免

### 数据集准备

从[这里](https://share-course.obs.cn-north-4.myhuaweicloud.com/dataset/cmn.zip)下载英中翻译所需要的数据集。文件说明如下所示：

- ch_en_all.txt:  英中翻译原始预料，共23607条，每条都是一句英文一句中文。
- ch_en_vocab.txt：翻译词表，包括中文词和英文词，已经句子分割词。

### 脚本准备

从[课程gitee仓库](https://gitee.com/mindspore/course)上下载本实验相关脚本。其中`tokenization.py`来源于[google-research/bert](https://github.com/google-research/bert/blob/master/tokenization.py)

### 上传文件

点击新建的OBS桶名，再打开“对象”标签页，通过“上传对象”、“新建文件夹”等功能，将脚本和数据集上传到OBS桶中，组织为如下形式：

```
transformer
└── data
│   ├── ch_en_all.txt
│   ├── ch_en_vocab.txt
└── code
       ├── bleu.ipynb
       ├── create_data.py
       ├── eval.py
       ├── train.py
       └── src
    	   └── 脚本等文件
```

## 实验步骤

实验步骤如下所示：

1. 运行create_data.py文件，生成训练测试需要的预处理文件。
2. 修改train_config.py配置文件，运行train.py，生成模型。
3. 修改eval_config.py配置文件，运行eval.py,生成预测结果。
4. 将预测结果文件传递给bleu.ipynb并运行，得到评测结果。

### 数据预处理

create_data.py代码是对原始数据进行处理。主要处理有两个方面：

1. 按照8：2的比例分割为训练数据和测试数据。
2. 对照词表ch_en_vocab.txt将句子转换为token（词转换为数字id）。并保存为mindrecord格式。

处理生成文件如下所示：

- source_train.txt: 训练数据集txt格式。包括中英对照。共18886条。
- source_test.txt:测试数据集txt格式。包括中英对照。共4721条。
- train.mindrecord：训练数据集mindrecord格式，本实验直接使用mindrecord格式数据训练
- test.mindrecord：测试数据集mindrecord格式，本实验直接使用mindrecord格式数据训练

### 模型代码梳理

代码文件说明：

- train.py：训练代码入口文件；
- eval.py：测试代码入口文件；
- train_config.py：训练配置文件；
- eval_config.py：测试配置文件；
- transformer_model.py：transformer模型文件；
- transformer_for_train.py: 训练loss文件。
- beam_search.py: transformer模型解码文件
- lr_schedule.py：学习率文件
- weight_init.py：模型权重初始化文件

#### transformer_model.py和beam_search.py代码梳理

`transformer_model.py`中`BertModel`接收数据输入。

训练网络梳理：

```
Class TransformerModel
- EmbeddingLookup
- EmbeddingPostprocessor
- CreateAttentionMaskFromInputMask
- TransformerEncoder  Encoder编码
- BeamSearchDecoder  
- PredLogProbs
```

测试网络梳理：
```
Class TransformerModel
- EmbeddingLookup
- EmbeddingPostprocessor
- CreateAttentionMaskFromInputMask
- TransformerEncoder  Encoder编码
- TileBeam
- BeamSearchDecoder
```

### 参数设定

预处理数据配置：create_data.py / cfg

```python
cfg = edict({
        'input_file': './data/ch_en_all.txt',
        'vocab_file': './data/ch_en_vocab.txt',
        'train_file_mindrecord': './path_cmn/train.mindrecord',
        'eval_file_mindrecord': './path_cmn/test.mindrecord',
        'train_file_source': './path_cmn/source_train.txt',
        'eval_file_source': './path_cmn/source_test.txt',
        'num_splits':1,
        'clip_to_max_len': False,
        'max_seq_length': 40
})
```

训练配置：train_config.py / cfg

```python
cfg = edict({
    #--------------------------------------nework confige---------------
    'transformer_network': 'base',
    'init_loss_scale_value': 1024,
    'scale_factor': 2,
    'scale_window': 2000,

    'lr_schedule': edict({
        'learning_rate': 1.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    }),
    #-----------------------------------save model confige-------------
    'enable_save_ckpt': True ,        #Enable save checkpointdefault is true.
    'save_checkpoint_steps':590,   #Save checkpoint steps, default is 590.
    'save_checkpoint_num':2,     #Save checkpoint numbers, default is 2.
    'save_checkpoint_path': './checkpoint',    #Save checkpoint file path,default is ./checkpoint/
    'save_checkpoint_name':'transformer-32_40',
    'checkpoint_path':'',     #Checkpoint file path
    
    
    #-------------------------------device confige----------------------
    'enable_data_sink':False,   #Enable data sink, default is False.
    'device_id':0,
    'device_num':1,
    'distribute':False,
    
    # -----------------mast same with the dataset-----------------------
    'seq_length':40,
    'vocab_size':10067,
    
    #-------------------------------------------------------------------
    'data_path':"./data/train.mindrecord",   #Data path
    'epoch_size':15,
    'batch_size':32,
    'max_position_embeddings':40,
    'enable_lossscale': False,  #Use lossscale or not, default is False.
    'do_shuffle':True     #Enable shuffle for dataset, default is True.
})
```

测试文件配置：eval_config.py / cfg

```python
cfg = edict({
    'transformer_network': 'base',
    
    'data_file': './data/test.mindrecord',
    'test_source_file':'./data/source_test.txt',
    'model_file': './ckpt/transformer-32_40-15_590.ckpt' ,
    'vocab_file':'./data1/ch_en_vocab.txt',
    'token_file': './token-32-40.txt',
    'pred_file':'./pred-32-40.txt',
    
    # ----------------mast same with the train config and the datsset---
    'seq_length':40,
    'vocab_size':10067,

    #-------------------------------------eval config-------------------
    'batch_size':32,
    'max_position_embeddings':40       # mast same with the train config
})

```
>**说明：** 
>
>1. 测试配置参数`seq_length`、`max_position_embeddings`、`vocab_size`必须与训练配置相同
>2. 参数`vocab_size`、`seq_length`与数据集统一。本实验给定的数据集`seq_length=40`（与source_train.txt、source_test.txt文件英/中最长长度相同）、`vocab_size=10067`(与vocab文件行数相同)
>3. 参数`max_position_embeddings`的值大于或等于`seq_length`

`schema_file` 文件时控制输入样本个数的。为.json格式，如下所示。其中input_ids、segment_ids、input_mask、label_ids代表四个输入字段。type、rank、shape代表各个字段的类型、开始rank、数据维度。

### 适配训练作业

创建训练作业时，运行参数会通过脚本传参的方式输入给脚本代码，脚本必须解析传参才能在代码中使用相应参数。如data_url和train_url，分别对应数据存储路径(OBS路径)和训练输出路径(OBS路径)。脚本对传参进行解析后赋值到args变量里，在后续代码里可以使用。（以测试为例）

```python
parser = argparse.ArgumentParser(description='Transformer testing')
    parser.add_argument('--data_url', required=True, default=None, help='Location of pre data.')   
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--ckpt_url', required=True, default=None, help='Location of model.') 
    parser.add_argument('--data_source', required=True, default=None, help='Location of source data.') 
args_opt = parser.parse_args()
```

MindSpore暂时没有提供直接访问OBS数据的接口，需要通过ModelArts自带的moxing框架与OBS交互。将OBS桶中的数据拷贝至执行容器中，供MindSpore使用：

```python
import moxing as mox
# src_url形如's3://OBS/PATH'，为OBS桶中数据集的路径，dst_url为执行容器中的路径
mox.file.copy_parallel(src_url=args_opt.data_url, dst_url='./data/')
mox.file.copy_parallel(src_url=args_opt.ckpt_url, dst_url='./ckpt/')
mox.file.copy_parallel(src_url=args_opt.data_source, dst_url='./data1/')
```

如需将训练输出（如模型Checkpoint）从执行容器拷贝至OBS，请参考：

```python
# src_url形如's3://OBS/PATH'，为OBS桶中数据集的路径，dst_url为执行容器中的路径
import moxing as mox
# src_url为执行容器中的路径，dst_url形如's3://OBS/PATH'，目录若不存在则会新建
mox.file.copy_parallel(src_url=cfg.token_file, dst_url=os.path.join(out_url,cfg.token_file))
mox.file.copy_parallel(src_url=cfg.pred_file, dst_url=os.path.join(out_url,cfg.pred_file))
```

### 创建训练作业

可以参考[使用常用框架训练模型](https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0238.html)来创建并启动训练作业。

打开[ModelArts控制台-训练管理-训练作业](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/trainingJobs)，点击“创建”按钮进入训练作业配置页面，创建训练作业的参考配置：

>- 算法来源：常用框架->Ascend-Powered-Engine->MindSpore；
>- 代码目录：选择上述新建的OBS桶中的transformer/code目录；
>- 启动文件：选择上述新建的OBS桶中的transformer/code目录下的`create_data.py`(预处理数据)，`train.py`（训练），`eval.py`（测试）；
>- 数据来源：数据存储位置->选择上述新建的OBS桶中的transformer目录下的data目录（预处理数据），data_pre目录(训练)，data_pre目录（测试）；
>- 训练输出位置：选择上述新建的OBS桶中的bert目录并在其中创建data_pre(预处理数据)，model目录（训练），eval_out目录（测试）；
>- 测试需要添加运行参数ckpt_url，设值为`s3://{user-obs}/transformer/model`。运行参数data_source，设值为`s3://{user-obs}/transformer/data`
>- 作业日志路径：同训练输出位置；
>- 规格：Ascend:1*Ascend 910；
>- 其他均为默认；

>**启动并查看训练过程：**
>
>1. 点击提交以开始训练；
>2. 在训练作业列表里可以看到刚创建的训练作业，在训练作业页面可以看到版本管理；
>3. 点击运行中的训练作业，在展开的窗口中可以查看作业配置信息，以及训练过程中的日志，日志会不断刷新，等训练作业完成后也可以下载日志到本地进行查看；
>4. 参考上述代码梳理，在日志中找到对应的打印信息，检查实验是否成功；
>5. 在日志中无错误且训练作业显示运行成功，即运行成功；

### 评测

本是按采用BLEU指标评测翻译结果。BLEU算法的思想就是机器翻译的译文越接近人工翻译的结果，它的翻译质量就越高。所以评测算法就是如何定义机器翻译译文与参考译文之间的相似度。

本实验采用自然语言工具包nltk中的sentence_bleu进行翻译结果评测（BLEU值计算）。评测代码见bleu.ipynb文件。

打开bleu.ipynb文件。如下所示

```
! pip install nltk   #安装nltk工具包

# --------------评测------------------
import moxing
moxing.file.copy_parallel(src_url='s3://{user-obs}/transformer/eval_out/', dst_url='./eval')
moxing.file.copy_parallel(src_url='s3://{user-obs}/transformer/data_pre/', dst_url='./train')

from nltk.translate.bleu_score import sentence_bleu

f = open('./eval/pred-32-40.txt', 'r', encoding='utf-8')
dic_rel={}
dic_pre={}
for line in f:
    line = line.strip().split('\t')
    if line[0] not in dic_rel:
        dic_rel[line[0]]=[line[1].split()]
    else:
        dic_rel[line[0]].append(line[1].split())
    if line[0] not in dic_pre:
        dic_pre[line[0]]=[line[2].split()]
    else:
        dic_pre[line[0]].append(line[2].split())
f.close()

f1 = open('./train/source_train.txt')
for l in f1:
    line = l.strip().split('\t')
    if line[0] in dic_rel:
        dic_rel[line[0]].append(line[1].split())
f1.close()

score=0
index=0
for en in dic_rel:  
    reference = dic_rel[en]
    print(reference)
    for candidate in dic_pre[en]:
        index += 1
        print(candidate)
        score_singe = sentence_bleu(reference, candidate)
        print(score_singe)
        score += score_singe
score_mean = score/index
print('BLUE_mean:',score_mean)
```
**说明：** 

1. `pred-32-40.txt`文件为测试输出预测结果。
2. `source_train.txt`文件为原始训练集文件。

## 实验结果

实验结果如下所示，只展示测试和评测结果。

测试翻译结果如下所示（pred.txt）,其中第一列为英文输入，第二列为标准翻译答案，第三列为模型预测输出。（测试输出保存了测试输出的token值和测试结果，这里不展示token值文件）

```
Hi .	你 好 。	嗨 ！
Run .	你 用 跑 的 。	拼 命 地 拼 命 地 址 。
I won !	我 赢 了 。	我 赢 得 了 ！
No way !	不 可 能 ！	没 有 路 ！
Try it .	试 试 吧 。	再 试 一 次 。
Why me ?	为 什 么 是 我 ？	为 什 么 我 ？
Be kind .	友 善 点 。	一 切 点 。
Get Tom .	找 到 汤 姆 。	汤 姆 走 了 。
Go home .	回 家 吧 。	回 家 。
Help me .	帮 我 一 下 。	随 时 帮 我 吃 吧 。
Hold on .	坚 持 。	停 下 来 。
Hug Tom .	抱 抱 汤 姆 ！	请 抱 歉 。
Shut up !	闭 嘴 ！	起 床 ！
...
...
She hit him .	她 打 了 他 。	她 被 他 撞 了 。
That ' s mine .	那 是 我 的 。	那 是 我 的 。
This is ice .	这 是 冰 块 。	这 太 冰 了 。
Tom got fat .	汤 姆 变 胖 了 。	汤 姆 胖 得 很 胖 。
Tom ' ll wait .	汤 姆 会 等 。	汤 姆 会 等 。
Tom ' s happy .	汤 姆 高 兴 。	汤 姆 很 快 乐 。
Wait for me .	等 等 我 。	等 我 。
We know him .	我 们 认 识 他 。	我 们 知 道 他 。
We want Tom .	我 们 想 要 汤 姆 。	我 们 想 要 汤 姆 。
What a pity !	太 可 惜 了 ！	多 遗 憾 啊 ！
What a pity !	多 遗 憾 啊 ！	多 遗 憾 啊 ！
Where ' s Tom ?	汤 姆 在 哪 儿 ？	汤 姆 在 哪 里 ？
You ' re mine .	你 是 我 的 。	你 是 我 的 。
You ' re sick !	你 有 病 ！	你 病 了 。
...
...
It ' s hard to believe that Tom wasn ' t aware that Mary was in love with him .	真 难 相 信 汤 姆 不 知 道 玛 丽 爱 他 。	很 难 相 信 汤 姆 是 不 知 道 玛 丽 爱 他 。
I don ' t mind lending you the money provided you pay it back within a month .	假 如 能 一 个 月 之 内 还 上 的 话 ， 我 可 以 把 钱 借 给 你 。	我 不 介 意 你 在 一 个 月 把 钱 接 受 一 下 来 的 钱 包 院 。
```

评测结果如下所示：

```
BLUE_mean:0.2540798544123031
```

## 结论

本实验主要介绍使用MindSpore实现Transformer网络，实现英中翻译任务。分析原理和结果可得：

- Transformer网络对翻译任务有效。
- Transformer网络测试采用循环测试（一个一个词测试），所以测试编译需要时间较长。
- 从结果分析，对于一词多义翻译较难。