# 使用指南

## 目录：

- Python Inference API
- Inference C++ API
- Inference实例
- Inference计算优化

## Python Inference API **[改进中]**
- 保存Inference模型 ([链接](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/io.py#L295))

  ```python
  def save_inference_model(dirname,
                           feeded_var_names,
                           target_vars,
                           executor,
                           main_program=None,
                           model_filename=None,
                           params_filename=None):
  ```
  Inference模型和参数将会保存到`dirname`目录下：
  - 序列化的模型
    - `model_filename`为`None`，保存到`dirname/__model__`
    - `model_filename`非`None`，保存到`dirname/model_filename`
  - 参数
    - `params_filename`为`None`，单独保存到各个独立的文件，各文件以参数变量的名字命名
    - `params_filename`非`None`，保存到`dirname/params_filename`

- 两种存储格式
  - 参数保存到各个独立的文件
    - 如，设置`model_filename`为`None`、`params_filename`为`None`

    ```bash
    $ cd recognize_digits_conv.inference.model
    $ ls
    $ __model__ batch_norm_1.w_0 batch_norm_1.w_2 conv2d_2.w_0 conv2d_3.w_0 fc_1.w_0 batch_norm_1.b_0 batch_norm_1.w_1 conv2d_2.b_0 conv2d_3.b_0 fc_1.b_0
    ```
  - 参数保存到同一个文件
    - 如，设置`model_filename`为`None`、`params_filename`为`__params__`

    ```bash
    $ cd recognize_digits_conv.inference.model
    $ ls
    $ __model__ __params__
    ```
- 加载Inference模型([链接](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/io.py#L380))
  ```python
  def load_inference_model(dirname,
                           executor,
                           model_filename=None,
                           params_filename=None):
    ...
    return [program, feed_target_names, fetch_targets]
  ```

## 链接Fluid Inference库
- 示例项目([链接](https://github.com/luotao1/fluid_inference_example.git))

  - GCC配置
    ```bash
    $ g++ -o a.out -std=c++11 main.cc \
          -I${PADDLE_ROOT}/ \
          -I${PADDLE_ROOT}/third_party/install/gflags/include \
          -I${PADDLE_ROOT}/third_party/install/glog/include \
          -I${PADDLE_ROOT}/third_party/install/protobuf/include \
          -I${PADDLE_ROOT}/third_party/eigen3 \
          -L${PADDLE_ROOT}/paddle/fluid/inference -lpaddle_fluid \
          -lrt -ldl -lpthread
    ```

  - CMake配置
    ```cmake
    include_directories(${PADDLE_ROOT}/)
    include_directories(${PADDLE_ROOT}/third_party/install/gflags/include)
    include_directories(${PADDLE_ROOT}/third_party/install/glog/include)
    include_directories(${PADDLE_ROOT}/third_party/install/protobuf/include)
    include_directories(${PADDLE_ROOT}/third_party/eigen3)
    target_link_libraries(${TARGET_NAME}
                          ${PADDLE_ROOT}/paddle/fluid/inference/libpaddle_fluid.so
                          -lrt -ldl -lpthread)
    ```

  - 设置环境变量：
  `export LD_LIBRARY_PATH=${PADDLE_ROOT}/paddle/fluid/inference:$LD_LIBRARY_PATH`



## C++ Inference API

- 推断流程([链接](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/test_helper.h#L91))

  - 1、 初始化设备
    ```cpp
    #include "paddle/fluid/framework/init.h"
    paddle::framework::InitDevices(false);
    ```

  - 2、 定义place，executor，scope
    ```cpp
    auto place = paddle::platform::CPUPlace();
    auto executor = paddle::framework::Executor(place);
    auto* scope = new paddle::framework::Scope();
    ```

  - 3、 加载模型
    ```cpp
    #include "paddle/fluid/inference/io.h"
    auto inference_program = paddle::inference::Load(executor, *scope, dirname);
    // or
    auto inference_program = paddle::inference::Load(executor,
                                                     *scope,
                                                     dirname + "/" + model_filename,
                                                     dirname + "/" + params_filename);
    ```

  - 4、 获取`feed_target_names`和`fetch_target_names`
    ```cpp
    const std::vector<std::string>& feed_target_names = inference_program->GetFeedTargetNames();
    const std::vector<std::string>& fetch_target_names = inference_program->GetFetchTargetNames();
    ```

  - 5、 准备`feed`数据
    ```cpp
    #include "paddle/fluid/framework/lod_tensor.h"
    std::vector<paddle::framework::LoDTensor*> cpu_feeds;
    ...
    std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
    for (size_t i = 0; i < feed_target_names.size(); ++i) {
      // Please make sure that cpu_feeds[i] is right for feed_target_names[i]
      feed_targets[feed_target_names[i]] = cpu_feeds[i];
    }
    ```

  - 6、 定义`Tensor`来`fetch`结果
    ```cpp
    std::vector<paddle::framework::LoDTensor*> cpu_fetchs;
    std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;
    for (size_t i = 0; i < fetch_target_names.size(); ++i) {
      fetch_targets[fetch_target_names[i]] = cpu_fetchs[i];
    }
    ```

  - 7、 执行`inference_program`
    ```cpp
    executor.Run(*inference_program, scope, feed_targets, fetch_targets);
    ```

  - 8、 使用`fetch`数据
    ```cpp
    for (size_t i = 0; i < cpu_fetchs.size(); ++i) {
      std::cout << "lod_i: " << cpu_fetchs[i]->lod();
      std::cout << "dims_i: " << cpu_fetchs[i]->dims();
      std::cout << "result:";
      float* output_ptr = cpu_fetchs[i]->data<float>();
      for (int j = 0; j < cpu_fetchs[i]->numel(); ++j) {
        std::cout << " " << output_ptr[j];
      }
      std::cout << std::endl;
    }
    ```
    针对不同的数据，4. - 8.可执行多次。

  - 9、 释放内存
    ```cpp
    delete scope;
    ```


- 接口说明

  ```cpp
  void Run(const ProgramDesc& program, Scope* scope,
           std::map<std::string, const LoDTensor*>& feed_targets,
           std::map<std::string, LoDTensor*>& fetch_targets,
           bool create_vars = true,
           const std::string& feed_holder_name = "feed",
           const std::string& fetch_holder_name = "fetch");
  ```
  - 使用Python API `save_inference_model`保存的`program`里面包含了`feed_op`和`fetch_op`，用户提供的`feed_targets`、`fetch_targets`必须和`inference_program`中的`feed_op`、`fetch_op`保持一致。
  - 用户提供的`feed_holder_name`和`fetch_holder_name`也必须和`inference_program`中`feed_op`、`fetch_op`保持一致，可使用`SetFeedHolderName`和`SetFetchHolderName`接口重新设置`inferece_program`
  - 默认情况下，除了`persistable`属性设置为`True`的`Variable`之外，每次执行`executor.Run`会创建一个局部`Scope`，并且在这个局部`Scope`中创建和销毁所有的`Variable`，以最小化空闲时的内存占用。
  - `persistable`属性为`True`的`Variable`有：
    - Operators的参数`w`、`b`等
    - `feed_op`的输入变量
    - `fetch_op`的输出变量


- **不在每次执行时创建和销毁变量
 ([PR](https://github.com/PaddlePaddle/Paddle/pull/9301))**
  - 执行`inference_program`
    ```cpp
    // Call once
    executor.CreateVariables(*inference_program, scope, 0);
    // Call as many times as you like
    executor.Run(
        *inference_program, scope, feed_targets, fetch_targets, false);
    ```
  - **优点**
    - 节省了频繁创建、销毁变量的时间（约占每次`Run`总时间的1% ~ 12%）
    - 执行结束后可获取所有Operators的计算结果
  - **缺点**
    - 空闲时也会占用大量的内存
    - 在同一个`Scope`中，相同的变量名是公用同一块内存的，容易引起意想不到的错误


- **不在每次执行时创建Op([PR](https://github.com/PaddlePaddle/Paddle/pull/9630))**
  - 执行`inference_program`
    ```cpp
    // Call once
    auto ctx = executor.Prepare(*inference_program, 0);
    // Call as many times as you like if you have no need to change the inference_program
    executor.RunPreparedContext(ctx.get(), scope, feed_targets, fetch_targets);
    ```
  - **优点**
    - 节省了频繁创建、销毁Op的时间
  - **缺点**
    - 一旦修改了`inference_program`，则需要重新创建`ctx`


- **多线程共享Parameters([链接](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/test_multi_thread_helper.h))**
  - 主线程
    - 1、 初始化设备
    - 2、 定义`place`，`executor`，`scope`
    - 3、 加载模型，得到`inference_program`
  - 从线程
    - **复制`inference_program`得到`copy_program`，修改`copy_program`的`feed_holder_name`和`fetch_holder_name`**
      ```cpp
      auto copy_program = std::unique_ptr<paddle::framework::ProgramDesc>(
                 new paddle::framework::ProgramDesc(*inference_program));
      std::string feed_holder_name = "feed_" + paddle::string::to_string(thread_id);
      std::string fetch_holder_name = "fetch_" + paddle::string::to_string(thread_id);
      copy_program->SetFeedHolderName(feed_holder_name);
      copy_program->SetFetchHolderName(fetch_holder_name);
      ```
    - 4、 获取`copy_program`的`feed_target_names`和`fetch_target_names`
    - 5、 准备feed数据，定义Tensor来fetch结果
    - 6、 执行`copy_program`
      ```cpp
      executor->Run(*copy_program, scope, feed_targets, fetch_targets, true, feed_holder_name, fetch_holder_name);
      ```
    - 7、 使用fetch数据
  - 主线程
    - 8、 释放资源


- 基本概念
  - 数据相关：
    - [Tensor](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/tensor.md)，一个N维数组，数据可以是任意类型（int，float，double等）
    - [LoDTensor](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/lod_tensor.md)，带LoD(Level-of-Detail)即序列信息的Tensor
    - [Scope](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/scope.md)，记录了变量Variable
  - 执行相关：
    - [Executor](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/executor.md)，无状态执行器，只跟设备相关
    - Place
      - CPUPlace，CPU设备
      - CUDAPlace，CUDA GPU设备
  - 神经网络表示：
    - [Program](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/program.md).

    详细介绍请参考[**Paddle Fluid开发者指南**](https://github.com/lcy-seso/learning_notes/blob/master/Fluid/developer's_guid_for_Fluid/Developer's_Guide_to_Paddle_Fluid.md)



## Inference实例

  1. fit a line: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_fit_a_line.cc)
  1. image classification: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_image_classification.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_image_classification.cc)
  1. label semantic roles: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_label_semantic_roles.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_label_semantic_roles.cc)
  1. recognize digits: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_recognize_digits.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_recognize_digits.cc)
  1. recommender system: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_recommender_system.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_recommender_system.cc)
  1. understand sentiment: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_understand_sentiment.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_understand_sentiment.cc)
  1. word2vec: [Python](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_word2vec.py), [C++](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/book/test_inference_word2vec.cc)


## Inference计算优化
- 使用Python推理优化工具([inference_transpiler](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/inference_transpiler.py))
  ```python
  class InferenceTranspiler:
    def transpile(self, program, place, scope=None):
        ...
        if scope is None:
            scope = global_scope()
        ...
  ```
  - 使用`InferenceTranspiler`将会直接修改`program`。
  - 使用`InferenceTranspiler`会修改参数的值，请确保`program`的参数在`scope`内。
- 支持的优化
  - 融合batch_norm op的计算
- 使用示例([链接](https://github.com/Xreki/Xreki.github.io/blob/master/fluid/inference/inference_transpiler.py))
  ```python
  import paddle.fluid as fluid
  # NOTE: Applying the inference transpiler will change the inference_program.
  t = fluid.InferenceTranspiler()
  t.transpile(inference_program, place, inference_scope)
  ```




## 内存使用优化
- 使用Python内存优化工具([memory_optimization_transipiler](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/memory_optimization_transpiler.py))
  ```python
  fluid.memory_optimize(inference_program)
  ```
