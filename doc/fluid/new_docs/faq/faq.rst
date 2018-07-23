###################
编译安装与单元测试
###################

1. 通过pip安装的PaddlePaddle在  :code:`import paddle.fluid` 报找不到 :code:`libmkldnn.so` 或 :code:`libmklml_intel.so`
------------------------------------------------------------------------------------------
出现这种问题的原因是在导入 :code:`paddle.fluid` 时需要加载 :code:`libmkldnn.so` 和 :code:`libmklml_intel.so`，
但是系统没有找到该文件。一般通过pip安装PaddlePaddle时会将 :code:`libmkldnn.so` 和 :code:`libmklml_intel.so`
拷贝到 :code:`/usr/local/lib` 路径下，所以解决办法是将该路径加到 :code:`LD_LIBRARY_PATH` 环境变量下，
即： :code:`export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH` 。

**注意**：如果是在虚拟环境中安装PaddlePaddle， :code:`libmkldnn.so` 和 :code:`libmklml_intel.so` 可能不在 :code:`/usr/local/lib` 路径下。
