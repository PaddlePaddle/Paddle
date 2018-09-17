## Install and Build

### Download & Install 

  Download the latest C-API development package from CI system and install. You can find the required version in the table below:
<table>
<thead>
<tr>
<th>Version Tips</th>
<th>C-API</th>
</tr>
</thead>
<tbody>
<tr>
<td>cpu_avx_mkl</td>
<td><a href="https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/paddle.tgz/?branch=0.14.0" rel="nofollow">paddle.tgz</a></td>
</tr>
<tr>
<td>cpu_avx_openblas</td>
<td><a href="https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/paddle.tgz/?branch=0.14.0" rel="nofollow">paddle.tgz</a></td>
</tr>
<tr>
<td>cpu_noavx_openblas</td>
<td><a href="https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuNoavxOpenblas/.lastSuccessful/paddle.tgz/?branch=0.14.0" rel="nofollow">paddle.tgz</a></td>
</tr>
<tr>
<td>cuda7.5_cudnn5_avx_mkl</td>
<td><a href="https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda75cudnn5cp27cp27mu/.lastSuccessful/paddle.tgz/?branch=0.14.0" rel="nofollow">paddle.tgz</a></td>
</tr>
<tr>
<td>cuda8.0_cudnn5_avx_mkl</td>
<td><a href="https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/paddle.tgz/?branch=0.14.0" rel="nofollow">paddle.tgz</a></td>
</tr>
<tr>
<td>cuda8.0_cudnn7_avx_mkl</td>
<td><a href="https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/paddle.tgz/?branch=0.14.0" rel="nofollow">paddle.tgz</a></td>
</tr>
<tr>
<td>cuda9.0_cudnn7_avx_mkl</td>
<td><a href="https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda90cudnn7avxMkl/.lastSuccessful/paddle.tgz/?branch=0.14.0" rel="nofollow">paddle.tgz</a></td>
</tr>
</tbody></table>

### From source

  Users can also compile the C-API library from PaddlePaddle source code by compiling with the following compilation options:
  
<table>
<thead>
<tr>
<th>Options</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>WITH_C_API</td>
<td>ON</td>
</tr>
<tr>
<td>WITH_PYTHON</td>
<td>OFF（recommended）</td>
</tr>
<tr>
<td>WITH_SWIG_PY</td>
<td>OFF（recommended）</td>
</tr>
<tr>
<td>WITH_GOLANG</td>
<td>OFF（recommended）</td>
</tr>
<tr>
<td>WITH_GPU</td>
<td>ON/OFF</td>
</tr>
<tr>
<td>WITH_MKL</td>
<td>ON/OFF</td>
</tr></tbody></table>

It is best to set up with recommended values to avoid linking with unnecessary libraries. Set other compilation options as you need.

Pull the latest following code snippet from github, and configure compilation options(replace PADDLE_ROOT with the installation path of the PaddlePaddle C-API inference library):

```shell
PADDLE_ROOT=/path/of/capi
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$PADDLE_ROOT \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      -DWITH_GOLANG=OFF \
      -DWITH_PYTHON=OFF \
      -DWITH_MKL=OFF \
      -DWITH_GPU=OFF  \
      ..
```

After running the above code to generate Makefile , run: `make && make install`.  After successful compilation, the dependencies required by C-API(includes: (1)PaddlePaddle inference library and header files; (2) Third-party libraries and header files) will be stored in the `PADDLE_ROOT` directory.

If the compilation is successful, see the following directory structure under `PADDLE_ROOT`(includes PaddlePaddle header files and libraries, and third-party libraries and header files(determined by the link methods if necessary)):

```text
├── include
│   └── paddle
│       ├── arguments.h
│       ├── capi.h
│       ├── capi_private.h
│       ├── config.h
│       ├── error.h
│       ├── gradient_machine.h
│       ├── main.h
│       ├── matrix.h
│       ├── paddle_capi.map
│       └── vector.h
├── lib
│   ├── libpaddle_capi_engine.a
│   ├── libpaddle_capi_layers.a
│   ├── libpaddle_capi_shared.so
│   └── libpaddle_capi_whole.a
└── third_party
    ├── gflags
    │   ├── include
    │   │   └── gflags
    │   │       ├── gflags_completions.h
    │   │       ├── gflags_declare.h
    │   │       ...
    │   └── lib
    │       └── libgflags.a
    ├── glog
    │   ├── include
    │   │   └── glog
    │   │       ├── config.h
    │   │       ...
    │   └── lib
    │       └── libglog.a
    ├── openblas
    │   ├── include
    │   │   ├── cblas.h
    │   │   ...
    │   └── lib
    │       ...
    ├── protobuf
    │   ├── include
    │   │   └── google
    │   │       └── protobuf
    │   │           ...
    │   └── lib
    │       └── libprotobuf-lite.a
    └── zlib
        ├── include
        │   ...
        └── lib
            ...

```

### Linking Description:

There are three kinds of linking methods:

1. Linking with dynamic library `libpaddle_capi_shared.so`（This way is much more convenient and easier, **Without special requirements, it is recommended**）, refer to the following：
    1. Compiling with CPU version and using `OpenBLAS`; only need to link one library named `libpaddle_capi_shared.so` to develop prediction program through C-API.
    1. Compiling with CPU version and using `MKL` lib, you need to link MKL library directly to develop prediction program through PaddlePaddle C-API, due to `MKL` has its own dynamic library.
    1. Compiling with GPU version, CUDA library will be loaded dynamically on prediction program run-time, and also set CUDA library to  `LD_LIBRARY_PATH` environment variable.

2. Linking with static library `libpaddle_capi_whole.a`，refer to the following：
    1. Specify `-Wl,--whole-archive` linking options.
    1. Explicitly link third-party libraries such as `gflags`、`glog`、`libz`、`protobuf` .etc, you can find them under `PADDLE_ROOT/third_party` directory.
    1. Use OpenBLAS library if compiling C-API，must explicitly link `libopenblas.a`.
    1. Use MKL when compiling C-API, must explicitly link MKL dynamic library.

3. Linking with static library `libpaddle_capi_layers.a` and `libpaddle_capi_engine.a`，refer to the following：
    1. This linking methods is mainly used for mobile prediction.
    1. Split `libpaddle_capi_whole.a` into two static linking library at least to reduce the size of linking libraries.
    1. Specify `-Wl,--whole-archive -lpaddle_capi_layers`  and  `-Wl,--no-whole-archive -lpaddle_capi_engine` for linking.
    1. The third-party dependencies need explicitly link same as method 2 above. 
