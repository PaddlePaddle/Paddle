# 为什么清理Paddle V2 API

目前Paddle的V2 API是一个实验性质的临时实现。我们使用这个V2 API实际上达成了Paddle API未来的样子。而当前这个API的实现非常粗糙，具有如下问题:

* 当前实现使用Python反射调用了Paddle的原生网络配置API
	* 这种代码很难继续维护，也很难添加自定义注释等等
* Paddle使用了SWIG暴露Paddle的C++ API
	* 使用SWIG有若干问题，请参考之前的讨论，[Why use C-API instead of SWIG](../multi_language_interface/00.why_plain_c.md).
	* 简要来说，使用SWIG暴露Paddle API将Paddle强绑定在Python语言上，并且生成的Python代码不能方便的增加注释，且必须使用C++ 风格命名Python API。
* V2 API『创造』了一个`Parameters`不存在于Paddle Core中的概念。这个概念需要转移到Paddle C++中。
	* Paddle不支持完全的『恢复训练』。`Parameters.save`只会保存所有参数值，不会保存动态学习率等优化信息。
* Paddle写一个新的Layer特别复杂。
	* Paddle使用非常简单的Protobuf，每写一个Layer都要修改Protobuf定义。缺乏扩展性。
	* Paddle解析用户配置的代码写在了Python端。这有如下几个问题:
		* 性能问题。Python构造Protobuf本身比较慢。
		* 由于Paddle更改了API的设计，但是也保留了向后兼容性。于是形成了多套API同时并存，且互相调用的关系。维护成本很高。
		* 无法完成多语言并存的需求。如果第三方开发者想支持Go语言，那配置解析的工作还需要重写一遍。

这些问题都是当前设计讨论的范畴，也是我们目前比较紧急的工作。

# 任务依赖性排序


* 重构Paddle的网络配置解析工作需要:
	* 因为使用Python解析网络配置特别慢，并且需要满足未来多语言接口的需求。我们需要把模型解析放到C/C++中实现。
	* 所以，Paddle网络配置解析依赖于Paddle 多语言接口。

* Paddle 多语言接口目前准备使用C API实现。但是:
	* Paddle V2 API中有一些在Python端『创造』的概念，例如『Parameters』。如果我们想一劳永逸的解决问题，需要把『Parameters』这个概念真正放到Paddle C++ Core中
	* 所以多语言接口依赖于将Parameters放入Paddle C++ Core中。

* 将Parameters放入Paddle C++ Core中需要对Paddle本身的Parameter类进行重构梳理，并且将单机多显卡多CPU的参数分发聚合逻辑转移至Parameters中。
	* 将Parameter类拆解开，将不同功能分别拆解到不同类中。例如，参数优化逻辑，参数保存载入逻辑，等等。
	* 简化Parameter类的实现。

# 整体目标

简化，减少Paddle的代码。拆分Paddle的耦合逻辑，让系统更易维护。

# 任务甘特图

前期任务还是将在API V2开发中的原型代码移入Paddle C++端。这需要先简化清理Paddle中Parameters，再将Parameters逻辑在C++端实现，特别需要实现多个拓扑结构share同一个Parameters的情况。

```text

 +---------------------------------------------------------+                +----------------------------------------------------+          +--------------------------------------------+
 |                                                         |                |                                                    |          |                                            |
 |                                                         |                |                                                    |          |                                            |
 |     Put "Parameters" Concept into Paddle C++ Core       +---------------->          Uses-C-API instead of SWIG API            +--------->+  Refactor Computation Graph Representation |
 |                                                         |                |                                                    |          |                                            |
 |                                                         |                |                                                    |          |                                            |
 +---------------------------------------------------------+                +----------------------------------------------------+          +--------------------------------------------+
         |
         |
         |          +----------------------------------------+
         |          |                                        |
         |          |  Clean paddle::Parameter Class         |
         |          |    * Extract functionalities into      |
         |          |      many classes.                     |
         +---------->      * Save/Load                       |
                    |      * Optimize                        |
                    |      * Randomize/Zerolize.             |
                    |      * Create/Destroy.                 |
                    |                                        |
                    +----------------------------------------+
                          |
                          |
                          |
                          |
                          |      +-----------------------------------------------+
                          |      |                                               |
                          |      |  Put Value Dispatch/Gradient Merge Logic into |
                          |      |  Parameters.                                  |
                          +----> |                                               |
                                 |    * Maybe introduce `nccl` will make code    |
                                 |      clear                                    |
                                 |                                               |
                                 +-----------------------------------------------+
                                   |
                                   |
                                   |
                                   |  +-----------------------------------------------+
                                   |  |                                               |
                                   |  | Share Parameters between many topologies.     |
                                   +> |                                               |
                                      |   * Not totally figure out what should be     |
                                      |     done right now.                           |
                                      |                                               |
                                      +-----------------------------------------------+


```
