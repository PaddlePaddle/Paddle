功能名称：为 Paddle Inference 添加新的前端语言

开始日期：2021-10-22

RFC PR：[PaddlePaddle#35977]()

GitHub Issue：[PaddlePaddle#35977](https://github.com/PaddlePaddle/Paddle/issues/35977)

# 任务说明
- 任务标题：为 Paddle Inference 添加新的前端语言

- 技术标签：Paddle Inference，Java

- 任务难度：困难

- 详细描述：Paddle Inference 当前支持五种编译语言 C++/Python/C/Go/R。Paddle Inference有 良好的解耦设计，使得我们可以较容易的支持 多种语言作为用户接口，自然也可以支持更多的语言作为用户接口。不管是 Java、Javascript、Swift、Ruby…… 只要你感兴趣，都可以完成这个任务。

  

# 实现目标
为Paddle Inference 添加前端语言 Java。通过Java调用Paddle Inference，作用于服务器端和云端，提供高性能的推理能力。使得Java程序员通过简单灵活的Java接口，20行代码完成Java后端测的Paddle部署。

实现Config、Tensor、Predictor等Java前端类，完成Paddle Inference的调用，实现Paddle部署。



# 开发流程和API文档设计
## Java 开发流程实例

Java代码调用Paddle Inference执行预测库仅需以下五步

1. 设置config信息

   ```
    Config config = new Config();
    config.setModel("resnet50/inference.pdmodel", "resnet50/inference.pdiparams");
   ```

2. 创建predictor

   ```
   Predictor predictor = Predictor.createPaddlePredictor(config);
   ```

3. 设置模型输入和输出 Tensor

   ```
   String[] inNames = predictor.getInputNames();
   Tensor inHandle = predictor.getInputHandle(inNames[0]);
   float[] inData = new float[1*3*224*224];
   for(int i=0; i<inData.length; i++){
   	inData[i] = (float) (i % 255);
   }
   inHandle.reshape(new int[]{1, 3, 224, 224});
   inHandle.copyFromCpu(inData);
   ```

4. 执行预测

   ```
   predictor.run();
   ```

5. 获得预测结果

   ```
   String[] outNames = predictor.getOutputNames();
   Tensor outHandle = predictor.getOutputHandle(outNames[0]);
   float[] outData = new float[numElements(outHandle.shape()];
   outHandle.copyToCpu(outData);
   ```

## Java API 文档

### Tensor

```
// 获取 Tensor 维度信息
// 参数：无
// 返回：int[] - 包含 Tensor 维度信息的int数组
public int[] shape();

// 设置 Tensor 维度信息
// 参数：int[] shape - 包含维度信息的int数组
// 返回：None
public void reshape(int[] shape);

// 获取 Tensor 名称
// 参数：无
// 返回：string - Tensor 名称
public string name();

// 获取 Tensor 数据类型
// 参数：无
// 返回：DataType - Tensor 数据类型
public DataType type();

// 设置 Tensor 数据
// 参数：float[] data - Tensor 数据
// 返回：None
public void copyFromCpu(float[] data);

// 获取 Tensor 数据
// 参数：float[] data - 用来存储 Tensor 数据
// 返回：None
public void copyToCpu(float[] data);
```

### Predictor

```
// 根据 Config 构建预测执行对象 Predictor
// 参数: Config config - 用于构建 Predictor 的配置信息
// 返回: Predictor
public Predictor createPaddlePredictor(Config config);

// 获取模型输入 Tensor 的数量
// 参数：无
// 返回：int - 模型输入 Tensor 的数量
public int getInputNum();

// 获取模型输出 Tensor 的数量
// 参数：无
// 返回：int - 模型输出 Tensor 的数量
public int getOutputNum();

// 获取输入 Tensor 名称
// 参数：无
// 返回：String[] - 输入 Tensor 名称
public String[] getInputNames();

// 获取输出 Tensor 名称
// 参数：无
// 返回：String[] - 输出 Tensor 名称
public String[] getOutputNames();

// 获取输入 handle
// 参数：String name - 输入handle名称
// 返回：Tensor - 输入 handle
public Tensor getInputHandle(String name);

// 获取输出 handle
// 参数：String name - 输出handle名称
// 返回：Tensor - 输出 handle
public Tensor getOutputHandle(String name);

// 执行预测
// 参数：无
// 返回：None
public void run()

// 释放中间Tensor
// 参数：None
// 返回：None
public void clearIntermediateTensor()

// 释放内存池中的所有临时 Tensor
// 参数：None
// 返回：None
public void tryShrinkMemory()
```

### Config
```
// 创建 Config 对象
// 参数：None
// 返回：Config
public Config Config();

// 判断当前 Config 是否有效
// 参数：None
// 返回：boolean - 当前 Config 是否有效
public boolean IsValid();

// 设置模型文件路径
// 参数：String modelDir - 模型文件夹路径
// 返回：None
public void setModelDir(String modelDir)

// 获取非combine模型的文件夹路径
// 参数：无
// 返回：String - 模型文件夹路径
public String modelDir();

// 设置 CPU Blas 库计算线程数
// 参数：int mathThreadsNum - 计算线程数
// 返回：None
public void setCpuMathLibraryNumThreads(mathThreadsNum int32)

// 获取 CPU Blas 库计算线程数
// 参数：无
// 返回：int - cpu 计算线程数
public int cpuMathLibraryNumThreads();

// 启用 MKLDNN 进行预测加速
// 参数：无
// 返回：None
public void enableMkldnn()

// 判断是否启用 MKLDNN
// 参数：无
// 返回：boolean - 是否启用 MKLDNN
public boolean MkldnnEnabled();

// 启用 MKLDNN BFLOAT16
// 参数：无
// 返回：None
public void enableMkldnnBfloat16();

// 判断是否启用 MKLDNN BFLOAT16
// 参数：无
// 返回：boolean - 是否启用 MKLDNN BFLOAT16
public boolean mkldnnBfloat16Enabled();

// 启用 GPU 进行预测
// 参数：int memorySize - 初始化分配的gpu显存，以MB为单位
//      int deviceId - 设备id
// 返回：None
public enableUseGpu(int memorySize, int deviceId);

// 禁用 GPU 进行预测
// 参数：无
// 返回：None
public void disableGpu();

// 判断是否启用 GPU 
// 参数：无
// 返回：boolean - 是否启用 GPU 
public boolean useGpu();

// 获取 GPU 的device id
// 参数：无
// 返回：int -  GPU 的device id
public int gpuDeviceId();

// 获取 GPU 的初始显存大小
// 参数：无
// 返回：int -  GPU 的初始的显存大小
public int memoryPoolInitSizeMb();

// 初始化显存占总显存的百分比
// 参数：无
// 返回：float - 初始的显存占总显存的百分比
public float fractionOfGpuMemoryForPool();

// 启用 IR 优化
// 参数：booleam x - 是否开启 IR 优化，默认打开
// 返回：None
public void switchIrOptim(boolean x);

// 判断是否开启 IR 优化 
// 参数：无
// 返回：boolean - 是否开启 IR 优化
public boolean irOptim();

// 设置是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件
// 参数：x - 是否打印 IR，默认关闭
// 返回：None
public void switchIrDebug(boolean x);

// 开启内存/显存复用，具体降低内存效果取决于模型结构
// 参数：无
// 返回：None
public void enableMemoryOptim()

// 判断是否开启内存/显存复用
// 参数：无
// 返回：bool - 是否开启内/显存复用
public boolean memoryOptimEnabled() bool

// 打开 Profile，运行结束后会打印所有 OP 的耗时占比。
// 参数：无
// 返回：None
public void enableProfile()

// 判断是否开启 Profile
// 参数：无
// 返回：boolean - 是否开启 Profile
public boolean profileEnabled();

// 去除 Paddle Inference 运行中的 LOG
// 参数：无
// 返回：None
public void disableGlogInfo();

// 返回config的配置信息
// 参数：None
// 返回：String - config配置信息
func String summary();
```

# 现有技术

### Java Native Interface

JNI 全称 Java Native Interface。Java本地方法接口，它是Java语言允许Java代码与C、C++代码交互的标准机制。在一个Java应用程序中，我们可以使用我们需要的C++类库，并且直接与Java代码交互，而且在可以被调用的C++程序内，反过来调用Java方法。

### Deep Java Library

Deep Java Library (DJL)是一个用于深度学习的开源、高级、与引擎无关的Java框架。对于Java开发人员来说，DJL易于入门和使用。DJL提供了与任何其他常规Java库一样的本地Java开发体验和功能。

DJL建立了一个模型库(ModelZoo)的概念，引入了来自于GluonCV, TorchHub, Keras 预训练模型等70多个模型。所有的模型都可以一键导入，用户只需要使用默认或者自己写的输入输出工具就可以实现轻松的推理。
