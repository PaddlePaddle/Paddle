# 安装paddle with cinn

## docker环境建立

### Nvidia GPU
参考: https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile-by-make.html#paddlepaddle

### Hygon DCU
```shell
# 拉取镜像
docker pull dongbaiyue/sycl:dtk23.04-ubuntu20.04-gcc8.2-dpcpp240119
# image主要环境：
# ubuntu20.04，自带gcc9.4
# dtk23.04 路径/opt/dtk-23.04
# gcc8.2 路径/opt/compiler/gcc8.2，用于编译paddle
# DPC++ 路径/opt/DPC++
# 启动容器，其中-v为挂载目录
docker run -it --name xxx -v /home/liuyi39/docker_space:/workspace \
  --shm-size=128G --network=host --workdir=/workspace \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  dongbaiyue/sycl:dtk23.04-ubuntu20.04-gcc8.2-dpcpp240119 /bin/bash
# 容器启动之后，在容器内运行rocm-smi命令检查DCU卡是否可以被正确识别
rocm-smi
# 查看sycl环境是否正确
sycl-ls
```

## 克隆源码

```shell
$ git clone https://github.com/DongBaiYue/Paddle.git
$ cd Paddle
$ git checkout sycl  # 切换到sycl分支编译
$ mkdir -p build && cd build #创建并进入build目录
```

当前先单独编译CINN，联合编译paddle with cinn可能有问题。

## 单独编译CINN

注：paddle未来可能不再支持。

```shell
# 设置编译选项
$ cp -r ../cmake ./
$ vim cmake/cinn/config.cmake
# Whether enable SYCL runtime
#
# Possible values:
# - ON: enable SYCL with cmake's auto search.
# - OFF: disable SYCL
# - /path/to/sycl: use specific path to sycl root
set(CINN_WITH_SYCL OFF)
# Whether enable ROCM runtime
#
# Possible values:
# - ON: enable ROCM with cmake's auto search.
# - OFF: disable ROCM
set(CINN_WITH_ROCM OFF)
# 设置使用gcc8.2版本编译
$ export CC="/opt/compiler/gcc8.2/bin/gcc"
$ export CXX="/opt/compiler/gcc8.2/bin/g++"
# 编译CINN，WITH_GPU选项控制是否启用CUDA后端
$ cmake .. -DCINN_ONLY=ON -DWITH_CINN=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DPY_VERSION=3.x # 设置python具体版本，如3.8
$ make -j
# 安装CINN。设置PYTHONPATH
$ vim ~/.bashrc
export PADDLE_PATH=/home/share/Paddle
export PYTHONPATH=$PADDLE_PATH/build/python:${PYTHONPATH}
```
### 安装不带CINN的Paddle
```
# 下载预编译的安装包
wget https://paddle-device.bj.bcebos.com/2.5.2/dcu/paddlepaddle_rocm-2.5.2-cp39-cp39-linux_x86_64.whl
# 安装
python3.9 -m pip install paddlepaddle_rocm-2.5.2-cp39-cp39-linux_x86_64.whl
```

## paddle with cinn联合编译，未验证可能出错。

cuda环境

```shell
# 设置 sycl
$ cp -r ../cmake ./
$ vim cmake/cinn/config.cmake
set(CINN_WITH_SYCL ON)
# 编译paddle
$ cmake .. -DWITH_CINN=ON -DWITH_GPU=ON -DPY_VERSION=3.8 -DWITH_TESTING=OFF -DWITH_DISTRIBUTE=OFF  -DCMAKE_BUILD_TYPE=Release
$ make -j 16 #大约需要一个多小时
# 安装paddle whl
$ python3.8 -m pip uninstall paddlepaddle_gpu -y #卸载已安装的paddle，若未安装跳过这步
$ python3.8 -m pip install Paddle/build/python/dist/*.whl #安装python/dist目录下的whl包
# 或者
export PADDLE_PATH=/home/share/Paddle
export PYTHONPATH=$PADDLE_PATH/build/python:${PYTHONPATH}
```

## CINN example code

```python
import numpy as np
from numpy import testing
from cinn import frontend,common
from cinn.common import SYCLTarget

A_data = np.random.random([1, 3, 224, 224]).astype("float32")
B_data = np.random.random([1, 3, 224, 224]).astype("float32")
res_data = np.maximum(np.add(A_data, B_data), 0)
print(res_data[0][0][0][:5])

def build_run(target:common.Target):
    print("Model running at ", target.arch)
    # Define the NetBuilder.
    builder = frontend.NetBuilder(name="network")

    # Define the input variables of the model
    a = builder.create_input(type=common.Float(32), shape=(1, 3, 224, 224), id_hint="A")
    b = builder.create_input(type=common.Float(32), shape=(1, 3, 224, 224), id_hint="B")

    # Build the model using NetBuilder API
    y = builder.add(a, b)
    res = builder.relu(y)

    # Specify target and generate the computation
    computation = frontend.Computation.build_and_compile(target, builder)
    computation.get_tensor("A").from_numpy(A_data, target)
    computation.get_tensor("B").from_numpy(B_data, target)
    computation.execute()
    res_tensor = computation.get_tensor(str(res))
    res_data_cinn = res_tensor.numpy(target)
    print(res_data_cinn[0][0][0][:5])
    testing.assert_almost_equal(res_data, res_data_cinn)

SYCL_target = common.SYCLTarget()
#SYCL_target = common.SYCLTarget(arch=common.Target.Arch.AMDGPU)
build_run(SYCL_target)
```




