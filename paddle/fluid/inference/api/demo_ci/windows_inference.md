# windows inference
本文介绍windows inference，目前只提供了静态编译，编译出paddle_fluid.lib，包含了除openblas.dll之外的所有第三方依赖库。

1. 下载最新的paddle_fluid.lib和openblas.dll，并把它们放在同一个目录下。

2. 准备预训练好的模型文件，例如models中的模型，可以将模型用safe_inference_model接口保存下来。将模型文件放到该目录下

3. 进入Paddle/paddle/fluid/inference/api/demo_ci目录，新建build目录，然后使用cmake生成vs2015的solution文件。
其中PADDLE_LIB是前面的paddle_fluid.lib对应文件夹, CUDA_LIB指定为x64格式下的cuda系统库目录文件夹。
```shell
 cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=ON -DWITH_MKL=OFF -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=inference_icnet -DPADDLE_LIB=D:\to_the_paddle_fluid.lib -DCUDA_LIB=D:\tools\v8.0\lib\x64
```
然后用vs2015打开对应的项目文件，注意使用静态链接 "/MT"，生成对应的exe。将openblas.dll放到exe所在目录。

4. 该exe即为项目生成文件，可绑定运行。

## FAQ
1. cmake需要您手动下载，并添加到系统路径里
2. 路径中的不要包含空格，例如发现CUDA_LIB路径是Program Files(x86)可能会出错。可以将CUDA拷贝到一个新位置。
