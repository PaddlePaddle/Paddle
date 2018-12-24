# JIT Kernel

结合函数模板和JIT生成需要的kernel函数。
这里的kernel是比Operator中kernel更小级别的算子单元，更侧重的是在不同硬件上的性能。可以有多重第三方库的实现，每种实现有自己的`UseMe`函数负责什么条件下可以被调用。
这里实现的函数可以非常细粒度的函数方法，比如Vector MUL， 也可以是一个复杂的逻辑比如LSTM等。复杂的逻辑也可以由自己的底层函数拼接而成。
目前仅支持CPU上的高性能计算。

## 目录结构

```txt
PaddlePaddle/Paddle/paddle/fluid/
├── ...
└── operators/
    ├── .../
    └── jit/
        ├── ...
        ├── gen/
        │   └── ...
        |── more/
        │   ├── ...
        │   ├── mkl/
        │   │   └── ...
        │   ├── mkldnn/
        │   │   └── ...
        │   ├── mix/
        │   │   └── ...
        │   ├── intrinsic/
        │   │   └── ...
        │   └── openblas/
        │       └── ...
        └── refer/
            └── ...
```

基本类的定义都放在根目录下，根目录下包括gen,more和refer三个目录。每个目录下都是一种或者多种实现，每种kernel算子都需要有reference的实现，用作单元测试的基准，其他的实现都是可选的。
- gen: 代表使用jit生成的code，需要依赖xbyak库。该实现最关心的就是性能。
- refer: 代表reference的实现，每种kernel算子都需要有在CPU上的reference的实现，他主要关心的算法逻辑的正确性。
- more: 下面可以放入跟多实现，可以包括mkl，mkldnn，intrinsic，openblas等，也可以是自身已有的kernel组合。

## 动态获取

提供一个`jit::Get`方法，根据kernel类别获取，每种实现都有自己的使用范围，根据范围动态和当前条件选择需要的kernel函数。

## 测试

- 逻辑测试
    所有实现都要与refer的code对比，需要满足精度要求， 包括float和double的数据类型
- 性能测试
    所有实现的性能对比，并且与最终的`jit::Get`方法对比，该方法拿到的性能需要在各种条件下都是最好的。

# 如何添加新的算子

- 在`KernelType` 中添加 `your_key` .
- 实现Reference 的逻辑，这个是必须是在CPU上的实现，并且不能依赖任何第三方库。实现后在`refer/CmakeLists.txt`中添加`USE_JITKERNEL_REFER(your_key)`来使用该kernel.
- (optional) 实现更多的算法在`more`目录下，可以依赖mkl，intrinsic或者mkldnn等第三方库。
- (optional) 实现基于Xbyak的生成code，在`gen`目下。 jitcode需要实现自己的`JitCodeCreator`，并注册在与refer相同的`KernelType`上。
- 必要时可以添加新的`KernelTuples`，可以参考`XYZNTuples`，新加的Attr类型需要特例化`JitCodeKey`方法。
- 在`test.cc`中添加unit test，至少需要测试`float`和`double`两种数据类型，如有必要需要支持额外的数据类型，比如`int8`的相关函数。
- 在`benchmark.cc`中添加相应的性能对比，同一种kernel需要对比所有实现，并且确保`jit::Get`得到的实现一直是速度最快的。

# 优点
- 统一的Get方法，接口简单。
- 同一套逻辑可以有多套实现，可以依赖多套第三方库，互不影响。
- 目录结构清晰，不会在某个文件中有多个宏定义，导致的可读性差问题。
- 优化方便，可以直接针对某种属性针对性优化，并不影响其他属性下的性能。
- 可以支持多种平台，包括Linux，Mac 和 Windows，至少可以保证每种平台都可以正常work。后期也可以针对不同平台有针对的优化。框架层面可以使用统一接口，不必关心底层实现。
