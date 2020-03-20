# Train with C++ inference API

What is C++ inference API and how to install it:

see: [PaddlePaddle Fluid 提供了 C++ API 来支持模型的部署上线](https://paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/deploy/inference/index_cn.html)

After downloading the source code of Paddle, you can build your own inference lib:

```shell
PADDLE_ROOT=./Paddle
cd Paddle
mkdir build
cd build
cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_PYTHON=OFF \
      -DWITH_MKL=OFF \
      -DWITH_GPU=OFF  \
      -DON_INFER=ON \
      ..
make
make inference_lib_dist
```

## IMDB task

see: [IMDB Dataset of 50K Movie Reviews | Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Quick Start

### prepare data

```shell
    wget https://fleet.bj.bcebos.com/text_classification_data.tar.gz
    tar -zxvf text_classification_data.tar.gz
```
### build

```shell
    mkdir build
    cd build
    rm -rf *
    PADDLE_LIB=path/to/Paddle/build/fluid_install_dir
    cmake .. -DPADDLE_LIB=$PADDLE_LIB  -DWITH_MKLDNN=OFF -DWITH_MKL=OFF
    make
```

### generate program description

```
    python generate_program.py bow
```

### run

```shell
   # After editing train.cfg
   sh run.sh
```

## results

Below are training logs on BOW model, the losses go down as expected.

```
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0731 22:39:06.974232 10965 demo_trainer.cc:130] Start training...
I0731 22:39:57.395229 10965 demo_trainer.cc:164] epoch: 0; average loss: 0.405706
I0731 22:40:50.262344 10965 demo_trainer.cc:164] epoch: 1; average loss: 0.110746
I0731 22:41:49.731079 10965 demo_trainer.cc:164] epoch: 2; average loss: 0.0475805
I0731 22:43:31.398355 10965 demo_trainer.cc:164] epoch: 3; average loss: 0.0233249
I0731 22:44:58.744391 10965 demo_trainer.cc:164] epoch: 4; average loss: 0.00701507
I0731 22:46:30.451735 10965 demo_trainer.cc:164] epoch: 5; average loss: 0.00258187
I0731 22:48:14.396687 10965 demo_trainer.cc:164] epoch: 6; average loss: 0.00113157
I0731 22:49:56.242744 10965 demo_trainer.cc:164] epoch: 7; average loss: 0.000698234
I0731 22:51:11.585919 10965 demo_trainer.cc:164] epoch: 8; average loss: 0.000510136
I0731 22:52:50.573947 10965 demo_trainer.cc:164] epoch: 9; average loss: 0.000400932
I0731 22:54:02.686152 10965 demo_trainer.cc:164] epoch: 10; average loss: 0.000329259
I0731 22:54:55.233342 10965 demo_trainer.cc:164] epoch: 11; average loss: 0.000278644
I0731 22:56:15.496256 10965 demo_trainer.cc:164] epoch: 12; average loss: 0.000241055
I0731 22:57:45.015926 10965 demo_trainer.cc:164] epoch: 13; average loss: 0.000212085
I0731 22:59:18.419997 10965 demo_trainer.cc:164] epoch: 14; average loss: 0.000189109
I0731 23:00:15.409077 10965 demo_trainer.cc:164] epoch: 15; average loss: 0.000170465
I0731 23:01:38.795770 10965 demo_trainer.cc:164] epoch: 16; average loss: 0.000155051
I0731 23:02:57.289487 10965 demo_trainer.cc:164] epoch: 17; average loss: 0.000142106
I0731 23:03:48.032507 10965 demo_trainer.cc:164] epoch: 18; average loss: 0.000131089
I0731 23:04:51.195230 10965 demo_trainer.cc:164] epoch: 19; average loss: 0.000121605
I0731 23:06:27.008040 10965 demo_trainer.cc:164] epoch: 20; average loss: 0.00011336
I0731 23:07:56.568284 10965 demo_trainer.cc:164] epoch: 21; average loss: 0.000106129
I0731 23:09:23.948290 10965 demo_trainer.cc:164] epoch: 22; average loss: 9.97393e-05
I0731 23:10:56.062590 10965 demo_trainer.cc:164] epoch: 23; average loss: 9.40532e-05
I0731 23:12:23.014047 10965 demo_trainer.cc:164] epoch: 24; average loss: 8.89622e-05
I0731 23:13:21.439818 10965 demo_trainer.cc:164] epoch: 25; average loss: 8.43784e-05
I0731 23:14:56.171597 10965 demo_trainer.cc:164] epoch: 26; average loss: 8.02322e-05
I0731 23:16:01.513542 10965 demo_trainer.cc:164] epoch: 27; average loss: 7.64629e-05
I0731 23:17:18.709139 10965 demo_trainer.cc:164] epoch: 28; average loss: 7.30239e-05
I0731 23:18:41.421555 10965 demo_trainer.cc:164] epoch: 29; average loss: 6.98716e-05
```

I trained a Bow model and a CNN model on IMDB dataset using the trainer. At the same time, I also trained the same models using traditional Python training methods. 
Results show that the two methods achieve almost the same dev accuracy:

CNN:
 
<img src="https://user-images.githubusercontent.com/23031310/62356234-32217300-b543-11e9-89fd-a07614904a08.png" width="300">

BOW:

<img src="https://user-images.githubusercontent.com/23031310/62356253-39488100-b543-11e9-9fa2-a399fc1119d6.png" width="300">

I also recorded the training speed of the C++ Trainer and the python training methods, C++ trainer is quicker on CNN model: 

<img src="https://user-images.githubusercontent.com/23031310/62356444-af4ce800-b543-11e9-88c8-f3bde1321ea1.png" width="300">

#TODO (mapingshuo): find the reason why C++ trainer is quicker on CNN model than python method.
