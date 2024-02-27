# 如何编译和使用cutlass的conv2d算子

本文件夹下面对应的是基于cutlass开发的conv2d算子，此算子被独立编译成so，供paddle内部的phi库调用。
这样做的好处有两个：
1. 减少paddle 发版时包的体积，避免把cutlass的代码打包进paddle inference中。
2. 将框架和算子具体实现完全解耦开，保证paddle框架的通用性的同时，保证具体算子实现的灵活性。

用户可以采用如下步骤编译和使用此算子

step1.

`bash compile.sh`

注意，该脚本中有三个参数需要用户自己指定下，分别是python解释器的路径，cuda的根目录路径和用户GPU机器的计算能力。
```shell
python_exe_path="python"
cuda_root_path="/usr/local/cuda"
gpu_cc="75"
```
compile.sh 脚本中会下载cutlass，执行CMakeLists.txt脚本，编译生成动态库。


step2.

step1执行后，就可以看到在 build 目录生成了 `libCutlassConv2d.so` ，并将build目录添加到LD_LIBRARY_PATH中即可使用此库。


step3.

默认情况下，在处理conv2d类算子时，Paddle Inference 会调用cuDNN实现；
基于 cutlass 开发的conv2d类算子能够融合更多的后处理算子，用户可以通过python API `exp_enable_use_cutlass()` 和 C++ API `Exp_EnableUseCutlass()`来获得一定的速度和显存收益。
