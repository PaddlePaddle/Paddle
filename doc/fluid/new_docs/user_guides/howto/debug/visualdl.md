# VisualDL (Visualize the Deep Learning)
<p align="center">
  <img src="https://raw.githubusercontent.com/PaddlePaddle/VisualDL/develop/docs/images/vs-logo.png" width="60%" />
</p>

## 介绍
VisualDL是一个面向深度学习任务设计的可视化工具，包含了scalar、参数分布、模型结构、图像可视化等功能，项目正处于高速迭代中，新的组件会不断加入。

目前大多数DNN平台均使用Python作为配置语言，VisualDL原生支持python的使用，
通过在模型的Python配置中添加几行，便可以为训练过程提供丰富的可视化支持。

除了Python SDK之外，VisualDL底层采用C++编写，其暴露的C++ SDK也可以集成到其他平台中，
实现原生的性能和定制效果。

## 组件
VisualDL 目前支持4种组件：

- graph
- scalar
- image
- histogram

### Graph
兼容 ONNX(Open Neural Network Exchange)[https://github.com/onnx/onnx], 通过与 python SDK的结合，VisualDL可以兼容包括 PaddlePaddle, pytorch, mxnet在内的大部分主流DNN平台。

<p align="center">
  <img src="https://raw.githubusercontent.com/daming-lu/large_files/master/graph_demo.gif" width="60%" />
</p>

### Scalar
可以用于展示训练测试的误差趋势

<p align="center">
<img src="https://raw.githubusercontent.com/daming-lu/large_files/master/loss_scalar.gif" width="60%"/>
</p>

### Image
可以用于可视化任何tensor，或模型生成的图片

<p align="center">
<img src="https://raw.githubusercontent.com/daming-lu/large_files/master/loss_image.gif" width="60%"/>
</p>

### Histogram

用于可视化任何tensor中元素分布的变化趋势

<p align="center">
<img src="https://raw.githubusercontent.com/daming-lu/large_files/master/histogram.gif" width="60%"/>
</p>

## 快速尝试
请使用下面的命令，来快速测试 VisualDL。

```
# 安装，建議是在虚拟环境或anaconda下。
pip install --upgrade visualdl

# 运行一个例子，vdl_create_scratch_log 将创建测试日志
vdl_create_scratch_log
visualDL --logdir=scratch_log --port=8080

# 访问 http://127.0.0.1:8080
```

如果以上步骤出现问题，很可能是因为python或pip不同版本或不同位置所致，以下安装方法能解决。

## 使用 virtualenv 安装

[Virtualenv](https://virtualenv.pypa.io/en/stable/) 能创建独立Python环境，也能确保Python和pip的相对位置正确。

在macOS上，安装pip和virtualenv如下：
```
sudo easy_install pip
pip install --upgrade virtualenv
```

在Linux上，安装pip和virtualenv如下:
```
sudo apt-get install python3-pip python3-dev python-virtualenv
```

然后创建一个虚拟环境：
```
virtualenv ~/vdl  # for Python2.7
virtualenv -p python3 ~/vdl for Python 3.x
```

```~/vdl``` 是你的Virtualenv目录, 你也可以选择任一目录。

激活虚拟环境如下：
```
source ~/vdl/bin/activate
```

现在再安装 VisualDL 和运行范例：

```
pip install --upgrade visualdl

# 运行一个例子，vdl_create_scratch_log 将创建测试日志
vdl_create_scratch_log
visualDL --logdir=scratch_log --port=8080

# 访问 http://127.0.0.1:8080
```
如果出现`TypeError: __init__() got an unexpected keyword argument 'file'`, 是因为protobuf不是3.5以上，运行`pip install --upgrade protobuf`就能解决。

如果在虚拟环境下仍然遇到安装问题，请尝试以下方法。


## 使用 Anaconda 安装

Anaconda是一个用于科学计算的Python发行版，提供了包管理与环境管理的功能，可以很方便地解决多版本python并存、切换以及各种第三方包安装问题。

请根据[Anaconda下载网站](https://www.anaconda.com/download) 的指示去下载和安装Anaconda.
下载Python 3.6版本的command-Line installer.

创建conda环境名字为```vdl```或任何名字:
```
conda create -n vdl pip python=2.7 # or python=3.3, etc.
```

激活conda环境如下:
```
source activate vdl
```

现在再安装 VisualDL 和运行范例：

```
pip install --upgrade visualdl

# 运行一个例子，vdl_create_scratch_log 将创建测试日志
vdl_create_scratch_log
visualDL --logdir=scratch_log --port=8080

# 访问 http://127.0.0.1:8080
```

如果仍然遇到安装问题，请尝试以下用源代码安装方法。

### 使用代码安装
```
#建議是在虚拟环境或anaconda下。
git clone https://github.com/PaddlePaddle/VisualDL.git
cd VisualDL

python setup.py bdist_wheel
pip install --upgrade dist/visualdl-*.whl
```

如果打包和安装遇到其他问题，不安装只想运行Visual DL可以看[这里](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/develop/how_to_dev_frontend_cn.md)


## SDK
VisualDL 同时提供了python SDK 和 C++ SDK 来实现不同方式的使用。

### Python SDK
VisualDL 现在支持 Python 2和 Python 3。

以最简单的Scalar组件为例，尝试创建一个scalar组件并插入多个时间步的数据：

```python
import random
from visualdl import LogWriter

logdir = "./tmp"
logger = LogWriter(logdir, sync_cycle=10000)

# mark the components with 'train' label.
with logger.mode("train"):
    # create a scalar component called 'scalars/scalar0'
    scalar0 = logger.scalar("scalars/scalar0")

# add some records during DL model running.
for step in range(100):
    scalar0.add_record(step, random.random())
```

### C++ SDK
上面 Python SDK 中代码完全一致的C++ SDK用法如下
```c++
#include <cstdlib>
#include <string>
#include "visualdl/sdk.h"

namespace vs = visualdl;
namespace cp = visualdl::components;

int main() {
  const std::string dir = "./tmp";
  vs::LogWriter logger(dir, 10000);

  logger.SetMode("train");
  auto tablet = logger.AddTablet("scalars/scalar0");

  cp::Scalar<float> scalar0(tablet);

  for (int step = 0; step < 1000; step++) {
    float v = (float)std::rand() / RAND_MAX;
    scalar0.AddRecord(step, v);
  }

  return 0;
}
```
## 启动Board
当训练过程中已经产生了日志数据，就可以启动board进行实时预览可视化信息

```
visualDL --logdir <some log dir>
```

board 还支持一下参数来实现远程的访问：

- `--host` 设定IP
- `--port` 设定端口
- `--model_pb` 指定 ONNX 格式的模型文件
