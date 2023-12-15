# 安装paddle with cinn

克隆源码

```shell
$ git clone https://github.com/DongBaiYue/Paddle.git
$ cd Paddle
$ git checkout sycl  # 切换到sycl分支编译
$ mkdir -p build && cd build #创建并进入build目录
```

可以单独编译CINN或联合编译paddle with cinn。

## 单独编译CINN

注：paddle未来可能不再支持。

```shell
# 设置 sycl
$ cp ../cmake ./
$ vim cmake/cinn/config.cmake
set(CINN_WITH_SYCL ON)
# 编译CINN
$ cmake .. -DCINN_ONLY=ON -DWITH_CINN=ON -DWITH_GPU=ON -DPY_VERSION=3.x # 设置python具体版本，如3.8
$ make -j
# 安装CINN。下面两步二选一，pip install或者设置PYTHONPATH
$ python3.8 -m pip install python/dist/cinn_gpu-xxx.whl #安装python/dist目录下的whl包
export PADDLE_PATH=/home/share/Paddle
export PYTHONPATH=$PADDLE_PATH/build/python:${PYTHONPATH}
# 安装不带cinn的paddle
$ python3.8 -m pip install paddlepaddle-gpu==0.0.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html #117为cuda版本
```

## paddle with cinn联合编译

cuda环境

```shell
# 设置 sycl
$ cp ../cmake ./
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

## sycl target
```
SYCL_target = common.SYCLTarget(arch=common.Target.Arch.NVGPU)
```




