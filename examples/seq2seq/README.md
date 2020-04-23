运行本目录下的范例模型需要安装PaddlePaddle Fluid 1.7版。如果您的 PaddlePaddle 安装版本低于此要求，请按照[安装文档](https://www.paddlepaddle.org.cn/#quick-start)中的说明更新 PaddlePaddle 安装版本。

# Sequence to Sequence (Seq2Seq)

以下是本范例模型的简要目录结构及说明：

```
.
├── README.md              # 文档，本文件
├── args.py                # 训练、预测以及模型参数配置程序
├── reader.py              # 数据读入程序
├── download.py            # 数据下载程序
├── train.py               # 训练主程序
├── predict.py             # 预测主程序
├── seq2seq_attn.py        # 带注意力机制的翻译模型程序
└── seq2seq_base.py        # 无注意力机制的翻译模型程序
```

## 简介

Sequence to Sequence (Seq2Seq)，使用编码器-解码器（Encoder-Decoder）结构，用编码器将源序列编码成vector，再用解码器将该vector解码为目标序列。Seq2Seq 广泛应用于机器翻译，自动对话机器人，文档摘要自动生成，图片描述自动生成等任务中。

本目录包含Seq2Seq的一个经典样例：机器翻译，实现了一个base model（不带attention机制），一个带attention机制的翻译模型。Seq2Seq翻译模型，模拟了人类在进行翻译类任务时的行为：先解析源语言，理解其含义，再根据该含义来写出目标语言的语句。更多关于机器翻译的具体原理和数学表达式，我们推荐参考飞桨官网[机器翻译案例](https://www.paddlepaddle.org.cn/documentation/docs/zh/user_guides/nlp_case/machine_translation/README.cn.html)。

## 模型概览

本模型中，在编码器方面，我们采用了基于LSTM的多层的RNN encoder；在解码器方面，我们使用了带注意力（Attention）机制的RNN decoder，并同时提供了一个不带注意力机制的解码器实现作为对比。在预测时我们使用柱搜索（beam search）算法来生成翻译的目标语句。

## 数据介绍

本教程使用[IWSLT'15 English-Vietnamese data ](https://nlp.stanford.edu/projects/nmt/)数据集中的英语到越南语的数据作为训练语料，tst2012的数据作为开发集，tst2013的数据作为测试集

### 数据获取

```
python download.py
```

## 模型训练

执行以下命令即可训练带有注意力机制的Seq2Seq机器翻译模型：

```sh
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --src_lang en --tar_lang vi \
    --attention True \
    --num_layers 2 \
    --hidden_size 512 \
    --src_vocab_size 17191 \
    --tar_vocab_size 7709 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --train_data_prefix data/en-vi/train \
    --eval_data_prefix data/en-vi/tst2012 \
    --test_data_prefix data/en-vi/tst2013 \
    --vocab_prefix data/en-vi/vocab \
    --use_gpu True \
    --model_path ./attention_models
```

可以通过修改 `attention` 参数为False来训练不带注意力机制的Seq2Seq模型，各参数的具体说明请参阅 `args.py` 。训练程序会在每个epoch训练结束之后，save一次模型。

默认使用动态图模式进行训练，可以通过设置 `eager_run` 参数为False来以静态图模式进行训练，如下：

```sh
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --src_lang en --tar_lang vi \
    --attention True \
    --num_layers 2 \
    --hidden_size 512 \
    --src_vocab_size 17191 \
    --tar_vocab_size 7709 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --train_data_prefix data/en-vi/train \
    --eval_data_prefix data/en-vi/tst2012 \
    --test_data_prefix data/en-vi/tst2013 \
    --vocab_prefix data/en-vi/vocab \
    --use_gpu True \
    --model_path ./attention_models \
    --eager_run False
```

## 模型预测

训练完成之后，可以使用保存的模型（由 `--reload_model` 指定）对test的数据集（由 `--infer_file` 指定）进行beam search解码，命令如下：

```sh
export CUDA_VISIBLE_DEVICES=0

python infer.py \
    --attention True \
    --src_lang en --tar_lang vi \
    --num_layers 2 \
    --hidden_size 512 \
    --src_vocab_size 17191 \
    --tar_vocab_size 7709 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --vocab_prefix data/en-vi/vocab \
    --infer_file data/en-vi/tst2013.en \
    --reload_model attention_models/10 \
    --infer_output_file infer_output.txt \
    --beam_size 10 \
    --use_gpu True
```

各参数的具体说明请参阅 `args.py` ，注意预测时所用模型超参数需和训练时一致。和训练类似，预测时同样可以以静态图模式进行，如下：

```sh
export CUDA_VISIBLE_DEVICES=0

python infer.py \
    --attention True \
    --src_lang en --tar_lang vi \
    --num_layers 2 \
    --hidden_size 512 \
    --src_vocab_size 17191 \
    --tar_vocab_size 7709 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --vocab_prefix data/en-vi/vocab \
    --infer_file data/en-vi/tst2013.en \
    --reload_model attention_models/10 \
    --infer_output_file infer_output.txt \
    --beam_size 10 \
    --use_gpu True \
    --eager_run False  
```

## 效果评价

使用 [*multi-bleu.perl*](https://github.com/moses-smt/mosesdecoder.git) 工具来评价模型预测的翻译质量，使用方法如下：

```sh
mosesdecoder/scripts/generic/multi-bleu.perl tst2013.vi < infer_output.txt
```

每个模型分别训练了10次，单次取第10个epoch保存的模型进行预测，取beam_size=10。效果如下（为了便于观察，对10次结果按照升序进行了排序）：

```
> no attention
tst2012 BLEU:
[10.75 10.85 10.9  10.94 10.97 11.01 11.01 11.04 11.13 11.4]
tst2013 BLEU:
[10.71 10.71 10.74 10.76 10.91 10.94 11.02 11.16 11.21 11.44]

> with attention
tst2012 BLEU:
[21.14 22.34 22.54 22.65 22.71 22.71 23.08 23.15 23.3  23.4]
tst2013 BLEU:
[23.41 24.79 25.11 25.12 25.19 25.24 25.39 25.61 25.61 25.63]
```
