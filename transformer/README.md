## Transformer

以下是本例的简要目录结构及说明：

```text
.
├── images               # README 文档中的图片
├── utils                # 工具包
├── gen_data.sh          # 数据生成脚本
├── predict.py           # 预测脚本
├── reader.py            # 数据读取接口
├── README.md            # 文档
├── train.py             # 训练脚本
├── model.py             # 模型定义文件
└── transformer.yaml     # 配置文件
```

## 模型简介

机器翻译（machine translation, MT）是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程，输入为源语言句子，输出为相应的目标语言的句子。

本项目是机器翻译领域主流模型 Transformer 的 PaddlePaddle 实现， 包含模型训练，预测以及使用自定义数据等内容。用户可以基于发布的内容搭建自己的翻译模型。


## 快速开始

### 安装说明

1. paddle安装

   本项目依赖于 PaddlePaddle 1.7及以上版本或适当的develop版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

2. 下载代码

    克隆代码库到本地
    ```shell
    git clone https://github.com/PaddlePaddle/models.git
    cd models/dygraph/transformer
    ```

3. 环境依赖

   请参考PaddlePaddle[安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/beginners_guide/install/index_cn.html)部分的内容


### 数据准备

公开数据集：WMT 翻译大赛是机器翻译领域最具权威的国际评测大赛，其中英德翻译任务提供了一个中等规模的数据集，这个数据集是较多论文中使用的数据集，也是 Transformer 论文中用到的一个数据集。我们也将[WMT'16 EN-DE 数据集](http://www.statmt.org/wmt16/translation-task.html)作为示例提供。运行 `gen_data.sh` 脚本进行 WMT'16 EN-DE 数据集的下载和预处理（时间较长，建议后台运行）。数据处理过程主要包括 Tokenize 和 [BPE 编码（byte-pair encoding）](https://arxiv.org/pdf/1508.07909)。运行成功后，将会生成文件夹 `gen_data`，其目录结构如下：

```text
.
├── wmt16_ende_data              # WMT16 英德翻译数据
├── wmt16_ende_data_bpe          # BPE 编码的 WMT16 英德翻译数据
├── mosesdecoder                 # Moses 机器翻译工具集，包含了 Tokenize、BLEU 评估等脚本
└── subword-nmt                  # BPE 编码的代码
```

另外我们也整理提供了一份处理好的 WMT'16 EN-DE 数据以供[下载](https://transformer-res.bj.bcebos.com/wmt16_ende_data_bpe_clean.tar.gz)使用，其中包含词典（`vocab_all.bpe.32000`文件）、训练所需的 BPE 数据（`train.tok.clean.bpe.32000.en-de`文件）、预测所需的 BPE 数据（`newstest2016.tok.bpe.32000.en-de`等文件）和相应的评估预测结果所需的 tokenize 数据（`newstest2016.tok.de`等文件）。


自定义数据：如果需要使用自定义数据，本项目程序中可直接支持的数据格式为制表符 \t 分隔的源语言和目标语言句子对，句子中的 token 之间使用空格分隔。提供以上格式的数据文件（可以分多个part，数据读取支持文件通配符）和相应的词典文件即可直接运行。

### 单机训练

### 单机单卡

以提供的英德翻译数据为例，可以执行以下命令进行模型训练：

```sh
# setting visible devices for training
export CUDA_VISIBLE_DEVICES=0

python -u train.py \
  --epoch 30 \
  --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
  --validation_file gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 4096
```

以上命令中传入了训练轮数（`epoch`）和训练数据文件路径（注意请正确设置，支持通配符）等参数，更多参数的使用以及支持的模型超参数可以参见 `transformer.yaml` 配置文件，其中默认提供了 Transformer base model 的配置，如需调整可以在配置文件中更改或通过命令行传入（命令行传入内容将覆盖配置文件中的设置）。可以通过以下命令来训练 Transformer 论文中的 big model：

```sh
# setting visible devices for training
export CUDA_VISIBLE_DEVICES=0

python -u train.py \
  --epoch 30 \
  --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
  --validation_file gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 4096 \
  --n_head 16 \
  --d_model 1024 \
  --d_inner_hid 4096 \
  --prepostprocess_dropout 0.3
```

另外，如果在执行训练时若提供了 `save_model`（默认为 trained_models），则每隔一定 iteration 后（通过参数 `save_step` 设置，默认为10000）将保存当前训练的到相应目录（会保存分别记录了模型参数和优化器状态的 `transformer.pdparams` 和 `transformer.pdopt` 两个文件），每隔一定数目的 iteration (通过参数 `print_step` 设置，默认为100)将打印如下的日志到标准输出：

```txt
[2019-08-02 15:30:51,656 INFO train.py:262] step_idx: 150100, epoch: 32, batch: 1364, avg loss: 2.880427, normalized loss: 1.504687, ppl: 17.821888, speed: 3.34 step/s
[2019-08-02 15:31:19,824 INFO train.py:262] step_idx: 150200, epoch: 32, batch: 1464, avg loss: 2.955965, normalized loss: 1.580225, ppl: 19.220257, speed: 3.55 step/s
[2019-08-02 15:31:48,151 INFO train.py:262] step_idx: 150300, epoch: 32, batch: 1564, avg loss: 2.951180, normalized loss: 1.575439, ppl: 19.128502, speed: 3.53 step/s
[2019-08-02 15:32:16,401 INFO train.py:262] step_idx: 150400, epoch: 32, batch: 1664, avg loss: 3.027281, normalized loss: 1.651540, ppl: 20.641024, speed: 3.54 step/s
[2019-08-02 15:32:44,764 INFO train.py:262] step_idx: 150500, epoch: 32, batch: 1764, avg loss: 3.069125, normalized loss: 1.693385, ppl: 21.523066, speed: 3.53 step/s
[2019-08-02 15:33:13,199 INFO train.py:262] step_idx: 150600, epoch: 32, batch: 1864, avg loss: 2.869379, normalized loss: 1.493639, ppl: 17.626074, speed: 3.52 step/s
[2019-08-02 15:33:41,601 INFO train.py:262] step_idx: 150700, epoch: 32, batch: 1964, avg loss: 2.980905, normalized loss: 1.605164, ppl: 19.705633, speed: 3.52 step/s
[2019-08-02 15:34:10,079 INFO train.py:262] step_idx: 150800, epoch: 32, batch: 2064, avg loss: 3.047716, normalized loss: 1.671976, ppl: 21.067181, speed: 3.51 step/s
[2019-08-02 15:34:38,598 INFO train.py:262] step_idx: 150900, epoch: 32, batch: 2164, avg loss: 2.956475, normalized loss: 1.580735, ppl: 19.230072, speed: 3.51 step/s
```

也可以使用 CPU 训练(通过参数 `--use_cuda False` 设置)，训练速度较慢。

#### 单机多卡

Paddle动态图支持多进程多卡进行模型训练，启动训练的方式如下：

```sh
python -m paddle.distributed.launch --started_port 8999 --selected_gpus=0,1,2,3,4,5,6,7 --log_dir ./mylog train.py \
  --epoch 30 \
  --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file gen_data/wmt16_ende_data_bpe/train.tok.clean.bpe.32000.en-de \
  --validation_file gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 4096 \
  --print_step 100 \
  --use_cuda True \
  --save_step 10000
```

此时，程序会将每个进程的输出log导入到`./mylog`路径下，只有第一个工作进程会保存模型。

```
.
├── mylog
│   ├── workerlog.0
│   ├── workerlog.1
│   ├── workerlog.2
│   ├── workerlog.3
│   ├── workerlog.4
│   ├── workerlog.5
│   ├── workerlog.6
│   └── workerlog.7
```

### 模型推断

以英德翻译数据为例，模型训练完成后可以执行以下命令对指定文件中的文本进行翻译：

```sh
# setting visible devices for prediction
export CUDA_VISIBLE_DEVICES=0

python -u predict.py \
  --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --predict_file gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 32 \
  --init_from_params trained_params/step_100000 \
  --beam_size 5 \
  --max_out_len 255 \
  --output_file predict.txt
```

 由 `predict_file` 指定的文件中文本的翻译结果会输出到 `output_file` 指定的文件。执行预测时需要设置 `init_from_params` 来给出模型所在目录，更多参数的使用可以在 `transformer.yaml` 文件中查阅注释说明并进行更改设置。注意若在执行预测时设置了模型超参数，应与模型训练时的设置一致，如若训练时使用 big model 的参数设置，则预测时对应类似如下命令：

```sh
# setting visible devices for prediction
export CUDA_VISIBLE_DEVICES=0

python -u predict.py \
  --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --predict_file gen_data/wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 32 \
  --init_from_params trained_params/step_100000 \
  --beam_size 5 \
  --max_out_len 255 \
  --output_file predict.txt \
  --n_head 16 \
  --d_model 1024 \
  --d_inner_hid 4096 \
  --prepostprocess_dropout 0.3
```


### 模型评估

预测结果中每行输出是对应行输入的得分最高的翻译，对于使用 BPE 的数据，预测出的翻译结果也将是 BPE 表示的数据，要还原成原始的数据（这里指 tokenize 后的数据）才能进行正确的评估。评估过程具体如下（BLEU 是翻译任务常用的自动评估方法指标）：

```sh
# 还原 predict.txt 中的预测结果为 tokenize 后的数据
sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt
# 若无 BLEU 评估工具，需先进行下载
# git clone https://github.com/moses-smt/mosesdecoder.git
# 以英德翻译 newstest2014 测试数据为例
perl gen_data/mosesdecoder/scripts/generic/multi-bleu.perl gen_data/wmt16_ende_data/newstest2014.tok.de < predict.tok.txt
```
可以看到类似如下的结果：
```
BLEU = 26.35, 57.7/32.1/20.0/13.0 (BP=1.000, ratio=1.013, hyp_len=63903, ref_len=63078)
```

使用本项目中提供的内容，英德翻译 base model 和 big model 八卡训练 100K 个 iteration 后测试有大约如下的 BLEU 值：

| 测试集 | newstest2014 | newstest2015 | newstest2016 |
|-|-|-|-|
| Base | 26.35 | 29.07 | 33.30 |
| Big | 27.07 | 30.09 | 34.38 |

### 预训练模型

我们这里提供了对应有以上 BLEU 值的 [base model](https://transformer-res.bj.bcebos.com/base_model_dygraph.tar.gz) 和 [big model](https://transformer-res.bj.bcebos.com/big_model_dygraph.tar.gz) 的模型参数提供下载使用（注意，模型使用了提供下载的数据进行训练和测试）。

## 进阶使用

### 背景介绍

Transformer 是论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中提出的用以完成机器翻译（machine translation, MT）等序列到序列（sequence to sequence, Seq2Seq）学习任务的一种全新网络结构，其完全使用注意力（Attention）机制来实现序列到序列的建模[1]。

相较于此前 Seq2Seq 模型中广泛使用的循环神经网络（Recurrent Neural Network, RNN），使用（Self）Attention 进行输入序列到输出序列的变换主要具有以下优势：

- 计算复杂度小
  - 特征维度为 d 、长度为 n 的序列，在 RNN 中计算复杂度为 `O(n * d * d)` （n 个时间步，每个时间步计算 d 维的矩阵向量乘法），在 Self-Attention 中计算复杂度为 `O(n * n * d)` （n 个时间步两两计算 d 维的向量点积或其他相关度函数），n 通常要小于 d 。
- 计算并行度高
  - RNN 中当前时间步的计算要依赖前一个时间步的计算结果；Self-Attention 中各时间步的计算只依赖输入不依赖之前时间步输出，各时间步可以完全并行。
- 容易学习长程依赖（long-range dependencies）
  - RNN 中相距为 n 的两个位置间的关联需要 n 步才能建立；Self-Attention 中任何两个位置都直接相连；路径越短信号传播越容易。

Transformer 中引入使用的基于 Self-Attention 的序列建模模块结构，已被广泛应用在 Bert [2]等语义表示模型中，取得了显著效果。


### 模型概览

Transformer 同样使用了 Seq2Seq 模型中典型的编码器-解码器（Encoder-Decoder）的框架结构，整体网络结构如图1所示。

<p align="center">
<img src="images/transformer_network.png" height=400 hspace='10'/> <br />
图 1. Transformer 网络结构图
</p>

可以看到，和以往 Seq2Seq 模型不同，Transformer 的 Encoder 和 Decoder 中不再使用 RNN 的结构。

### 模型特点

Transformer 中的 Encoder 由若干相同的 layer 堆叠组成，每个 layer 主要由多头注意力（Multi-Head Attention）和全连接的前馈（Feed-Forward）网络这两个 sub-layer 构成。
- Multi-Head Attention 在这里用于实现 Self-Attention，相比于简单的 Attention 机制，其将输入进行多路线性变换后分别计算 Attention 的结果，并将所有结果拼接后再次进行线性变换作为输出。参见图2，其中 Attention 使用的是点积（Dot-Product），并在点积后进行了 scale 的处理以避免因点积结果过大进入 softmax 的饱和区域。
- Feed-Forward 网络会对序列中的每个位置进行相同的计算（Position-wise），其采用的是两次线性变换中间加以 ReLU 激活的结构。

此外，每个 sub-layer 后还施以 Residual Connection [3]和 Layer Normalization [4]来促进梯度传播和模型收敛。

<p align="center">
<img src="images/multi_head_attention.png" height=300 hspace='10'/> <br />
图 2. Multi-Head Attention
</p>

Decoder 具有和 Encoder 类似的结构，只是相比于组成 Encoder 的 layer ，在组成 Decoder 的 layer 中还多了一个 Multi-Head Attention 的 sub-layer 来实现对 Encoder 输出的 Attention，这个 Encoder-Decoder Attention 在其他 Seq2Seq 模型中也是存在的。

## FAQ

**Q:** 预测结果中样本数少于输入的样本数是什么原因  
**A:** 若样本中最大长度超过 `transformer.yaml` 中 `max_length` 的默认设置，请注意运行时增大 `--max_length` 的设置，否则超长样本将被过滤。

**Q:** 预测时最大长度超过了训练时的最大长度怎么办  
**A:** 由于训练时 `max_length` 的设置决定了保存模型 position encoding 的大小，若预测时长度超过 `max_length`，请调大该值，会重新生成更大的 position encoding 表。


## 参考文献
1. Vaswani A, Shazeer N, Parmar N, et al. [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)[C]//Advances in Neural Information Processing Systems. 2017: 6000-6010.
2. Devlin J, Chang M W, Lee K, et al. [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805)[J]. arXiv preprint arXiv:1810.04805, 2018.
3. He K, Zhang X, Ren S, et al. [Deep residual learning for image recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
4. Ba J L, Kiros J R, Hinton G E. [Layer normalization](https://arxiv.org/pdf/1607.06450.pdf)[J]. arXiv preprint arXiv:1607.06450, 2016.
5. Sennrich R, Haddow B, Birch A. [Neural machine translation of rare words with subword units](https://arxiv.org/pdf/1508.07909)[J]. arXiv preprint arXiv:1508.07909, 2015.


## 作者
- [guochengCS](https://github.com/guoshengCS)

## 如何贡献代码

如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
