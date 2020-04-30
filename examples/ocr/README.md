简介
--------
本OCR任务是识别图片单行的字母信息，基于attention的seq2seq结构。 运行本目录下的程序示例需要使用PaddlePaddle develop最新版本。

## 代码结构
```
.
|-- data.py          # 数据读取
|-- eval.py          # 评估脚本
|-- images           # 测试图片
|-- predict.py       # 预测脚本
|-- seq2seq_attn.py  # 模型
|-- train.py         # 训练脚本
`-- utility.py       # 公共模块
```

## 训练/评估/预测流程

- 设置GPU环境:

```
export CUDA_VISIBLE_DEVICES=0
```

- 训练

```
python train.py
```

更多参数可以通过`--help`查看。


- 动静切换


```
python train.py --dynamic=True
```


- 评估

```
python eval.py --init_model=checkpoint/final
```


- 预测

目前不支持动态图预测

```
python predict.py --init_model=checkpoint/final --image_path=images/ --dynamic=False --beam_size=3
```

预测结果如下:

```
Image 1: images/112_chubbiness_13557.jpg
0: chubbines
1: chubbiness
2: chubbinesS
Image 2: images/177_Interfiled_40185.jpg
0: Interflied
1: Interfiled
2: InterfIled
Image 3: images/325_dame_19109.jpg
0: da
1: damo
2: dame
Image 4: images/368_fixtures_29232.jpg
0: firtures
1: Firtures
2: fixtures
```
