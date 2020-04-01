# 数据使用说明

## Kinetics数据集

Kinetics数据集是DeepMind公开的大规模视频动作识别数据集，有Kinetics400与Kinetics600两个版本。这里使用Kinetics400数据集，具体的数据预处理过程如下。

### mp4视频下载
在Code\_Root目录下创建文件夹

    cd $Code_Root/data/dataset && mkdir kinetics

    cd kinetics && mkdir data_k400 && cd data_k400

    mkdir train_mp4 && mkdir val_mp4

ActivityNet官方提供了Kinetics的下载工具，具体参考其[官方repo ](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)即可下载Kinetics400的mp4视频集合。将kinetics400的训练与验证集合分别下载到data/dataset/kinetics/data\_k400/train\_mp4与data/dataset/kinetics/data\_k400/val\_mp4。

### mp4文件预处理

为提高数据读取速度，提前将mp4文件解帧并打pickle包，dataloader从视频的pkl文件中读取数据（该方法耗费更多存储空间）。pkl文件里打包的内容为(video-id, label, [frame1, frame2,...,frameN])。

在 data/dataset/kinetics/data\_k400目录下创建目录train\_pkl和val\_pkl

    cd $Code_Root/data/dataset/kinetics/data_k400

    mkdir train_pkl && mkdir val_pkl

进入$Code\_Root/data/dataset/kinetics目录，使用video2pkl.py脚本进行数据转化。首先需要下载[train](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics/data/kinetics-400_train.csv)和[validation](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics/data/kinetics-400_val.csv)数据集的文件列表。

首先生成预处理需要的数据集标签文件

    python generate_label.py kinetics-400_train.csv kinetics400_label.txt

然后执行如下程序：

    python video2pkl.py kinetics-400_train.csv $Source_dir $Target_dir  8 #以8个进程为例

- 该脚本依赖`ffmpeg`库，请预先安装`ffmpeg`

对于train数据，

    Source_dir = $Code_Root/data/dataset/kinetics/data_k400/train_mp4

    Target_dir = $Code_Root/data/dataset/kinetics/data_k400/train_pkl

对于val数据，

    Source_dir = $Code_Root/data/dataset/kinetics/data_k400/val_mp4

    Target_dir = $Code_Root/data/dataset/kinetics/data_k400/val_pkl

这样即可将mp4文件解码并保存为pkl文件。

### 生成训练和验证集list
··
    cd $Code_Root/data/dataset/kinetics

    ls $Code_Root/data/dataset/kinetics/data_k400/train_pkl/* > train.list

    ls $Code_Root/data/dataset/kinetics/data_k400/val_pkl/* > val.list

    ls $Code_Root/data/dataset/kinetics/data_k400/val_pkl/* > test.list

    ls $Code_Root/data/dataset/kinetics/data_k400/val_pkl/* > infer.list

即可生成相应的文件列表，train.list和val.list的每一行表示一个pkl文件的绝对路径，示例如下：

    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/train_pkl/data_batch_100-097
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/train_pkl/data_batch_100-114
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/train_pkl/data_batch_100-118
    ...

或者

    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/val_pkl/data_batch_102-085
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/val_pkl/data_batch_102-086
    /ssd1/user/models/PaddleCV/PaddleVideo/data/dataset/kinetics/data_k400/val_pkl/data_batch_102-090
    ...
