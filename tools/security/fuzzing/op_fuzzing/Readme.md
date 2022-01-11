# Paddle op_fuzzing

此项目提供用例来说明使用libfuzzer+atheris对Paddle进行op kernel的fuzzing方法。

## 环境配置

### 1. 使用docker

使用目录中的Dockerfile进行环境构建。Dockerfile提供的环境包括：

- python3.8
- clang-12
- atheris
- cmake-3.16
- numpy
- protobuf

如Dockerfile所示，目录中所提供的patch针对Paddle版本为：
develop(249081b6ee9ada225c2aa3779a6935be65bc04e0)

### 2. 手动配置

自己手动配置编译及运行环境。需要安装的依赖有：

- python3.6-3.9
- clang-8 以上
- atheris 需要使用安装好的clang来编译
- cmake 需要3.16或以上
- numpy
- protobuf

Python环境在本机上最好创建一个新的虚拟环境用于构建Paddle。

`python3.8 -m venv paddle-env`

需要设置环境变量：

```shell
export PYTHON_EXECUTABLE=paddle-env/bin/python
export PYTHON_INCLUDE_DIRS=paddle-env/include
export PYTHON_LIBRARY=paddle-env/lib
```

选择合适的目标Paddle版本和commit，改动如下文件内容：

- cmake/flags.cmake

将`COMMON_FLAGS`中的`-Werror`改为`-Wno-error`以避免编译过程被warning打断。

- cmake/external/pybind11.cmake

`include_directories(/usr/include/python3.x)`添加include路径。

- paddle/fluid/platform/init.cc

注释代码：

```c++
google::InstallFailureSignalHandler();
google::InstallFailureWriter(&SignalHandle);
```

使其不会干扰到ASAN打印日志。

#### 编译及安装：

CC和CXX需要使用clang，且clang需与编译atheris的clang版本相同。

CPU版本（以Python3.8为例）：

```shell
mkdir build && cd build
CC=clang CXX=clang++ cmake .. -DPY_VERSION=3.8 -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DSANITIZER_TYPE=Address -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_AVX=OFF -DWITH_MKL=OFF
python3.8 -m pip install -U python/dist/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl
```

注意这里使用`-DSANITIZER_TYPE`指定了ASAN，也可以指定为其他的sanitizer，具体可参见Paddle cmake文件。

## Fuzzing

**fuzz_util.py**是一个fuzz通用结构模块，其使用到atheris的LDP接口来变异和抽象tensor，shape以及param_attr等数据结构。

每一个op都可以单独或者结合使用，写成fuzz用例，如项目目录中的用例所示。用例使用到atheris的Fuzz接口，结合libfuzzer来实现覆盖率指导fuzzing。

运行fuzzer需要设置`LD_PRELOAD`环境变量:

```shell
LD_PRELOAD="$(python -c "import atheris; import os; print(os.path.dirname(atheris.path()))")/asan_with_fuzzer.so"
```

运行fuzzer方法与libfuzzer相同，可参考：
[libfuzzer](https://llvm.org/docs/LibFuzzer.html)
和
[atheris](https://github.com/google/atheris)
官方使用文档。

- 有关Python代码部分的问题认同

Paddle未捕获的IndexError，ZeroDivisionError等问题均应属于bug，将会造成Paddle的不稳定性。相反，Paddle意识到且对输入捕获处理而抛出的如ValueError，RuntimeError等不属于bug。
