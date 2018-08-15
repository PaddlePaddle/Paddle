# Example
Anakin目前只支持NCHW的格式
示例文件在test/framework/net下

## 在NV的GPU上运行CNN模型
示例文件为打开example_nv_cnn_net.cpp，整体流程如下：
- 将模型的的path设置为anakin模型的路径，初始化NV平台的图对象。 anakin模型可以通过转换器转化caffe或fluid的模型得到
- 根据模型设置网络图的输入尺寸，进行图优化
- 根据优化后的网络图初始化网络执行器
- 取出网络的输入tensor，将数据拷贝到输入tensor
- 运行推导
- 取出网络的输出tensor

以NV平台为例演示Anakin框架的使用方法，注意编译时需要打开GPU编译开关

## 在X86上运行RNN模型
示例文件为example_x86_rnn_net.cpp
整体流程与在NV的GPU上运行CNN模型相似，不同之处如下：
- 使用X86标识初始化图对象和网络执行器对象
- rnn模型的输入尺寸是可变的，初始化图时的输入维度是维度的最大值，输入维度N代表总的词的个数。还需要设置输入tensor的seq_offset来标示这些词是如何划分为句子的,如{0,5,12}表示共有12个词，其中第0到第4个词是第一句话，第5到第11个词是第二句话

以X86平台为例演示Anakin框架的使用方法，注意编译时需要打开X86编译开关

## 在NV的GPU上使用Anakin的线程池运行CNN模型
示例文件为example_nv_cnn_net_multi_thread.cpp ，示例使用worker的同步预测接口
整体流程与在NV的GPU上运行CNN模型相似，不同之处如下：
- 用模型地址和线程池大小初始化worker对象
- 将输入tensor注入任务队列,获得输出tensor
