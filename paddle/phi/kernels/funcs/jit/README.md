# JIT Kernel

结合函数模板和JIT生成需要的kernel函数。
这里的kernel是比Operator中kernel更小级别的算子单元，更侧重的是在不同硬件上的性能。可以有多重第三方库的实现，每种实现有自己的`CanBeUsed`函数负责什么条件下可以被调用。
这里实现的函数可以非常细粒度的函数方法，比如Vector MUL， 也可以是一个复杂的逻辑比如LSTM等。复杂的逻辑也可以由自己的底层函数拼接而成。
目前仅支持CPU上的高性能计算。

## 目录结构

```txt
PaddlePaddle/Paddle/paddle/phi/kernels/
├── ...
└── funcs/
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
- more: 下面可以放入更多实现，可以包括mkl，mkldnn，intrinsic，openblas等，也可以是自身已有的kernel组合。

## 动态获取

- 提供`GetAllCandidateFuncs`方法，根据输入的kernel类别，获取满足要求的所有函数实现。所有实现保证结果一致，但是速度不一致，可以根据具体输入属性大小，动态测试得到当前最优实现，手动选择最优函数。
- 提供`GetDefaultBestFunc`方法，返回一个默认最优的函数实现。该函数是根据一些通用配置离线tuning之后的结果，能覆盖大多数情况下最优结果。
- 提供`KernelFuncs::Cache()`方法，该方法会返回默认最优的函数，同时会缓存该函数指针，如果出现属性一致的情况，直接返回上次的函数指针，如果不存在则根据属性新建。
- 提供`GetReferFunc` 方法，返回该kernel最原始的逻辑函数。该方法与kernel的输入大小和属性没有任何关系，有且并只有一个在CPU上的实现。该方法表征了kernel的原始逻辑，其他所有实现的逻辑与它保持一致。

### 例子

所有kernel的调用只需要在头文件中包含`"paddle/phi/kernels/funcs/jit/kernels.h"`， 该文件是编译时自动生成的。

直接从缓存中获取默认最优的函数。

```cpp
    using T = float;
    jit::seq_pool_attr_t attr(width, jit::SeqPoolType::kSum);
    auto seqpool_func = jit::KernelFuncs<jit::SeqPoolTuple<T>, phi::CPUPlace>::Cache().At(attr);
    seqpool_func(src_data, dst_data, &attr);
```

跑一遍所有实现，并输出实现类别。

```cpp
    using T = float;
    jit::seq_pool_attr_t attr(width, jit::SeqPoolType::kSum);
    auto funcs = jit::GetAllCandidateFuncsWithTypes<jit::SeqPoolTuple<T>, phi::CPUPlace>(attr);
    for (auto f : funcs) {
        LOG(INFO) << "Kernel implementation type: " << f.first;
        f.second(src_data, dst_data, &attr);
    }
```

## 测试

- 逻辑测试
    所有实现都要与refer的code对比，需要满足精度要求， 包括float和double的数据类型
- 性能测试
    所有实现的性能对比，并且与最终的`jit::GetDefaultBestFunc`方法对比，该方法拿到的性能需要在各种条件下都是最好的。

# 如何添加新的算子

1. 在`KernelType` 中添加 `your_key` 。
2. 实现Reference 的逻辑，这个是必须是在CPU上的实现，并且不能依赖任何第三方库。实现后在`refer/CMakeLists.txt`中添加`USE_JITKERNEL_REFER(your_key)`来使用该kernel。
3. (optional) 实现更多的算法在`more`目录下，可以依赖mkl，intrinsic或者mkldnn等第三方库。
4. (optional) 实现基于Xbyak的生成code，在`gen`目下。 jitcode需要实现自己的`JitCodeCreator`，并注册在与refer相同的`KernelType`上。
5. 添加新的`KernelTuple`，需要与`KernelType`一一对应，是所有类型的一个打包，包括数据类型，属性的类型，以及返回的函数类型。可以参考`SeqPoolTuple`，新加的Attr类型需要特例化`JitCodeKey`方法。
6. 在`test.cc`中添加unit test，至少需要测试`float`和`double`两种数据类型，如有必要需要支持额外的数据类型，比如`int8`的相关函数。
7. 在`benchmark.cc`中添加相应的性能对比，同一种kernel需要对比所有实现，并且确保`GetDefaultBestFunc`得到的实现一直是速度最快的。

# 优点
- 接口方便，灵活调用。
- 同一套逻辑可以有多套实现，可以依赖多套第三方库，互不影响。
- 目录结构清晰，不会在某个文件中有多个宏定义，导致的可读性差问题。
- 优化方便，可以直接针对某种属性针对性优化，并不影响其他属性下的性能。
- 可以支持多种平台，包括Linux，Mac 和 Windows，至少可以保证每种平台都可以正常work。后期也可以针对不同平台有针对的优化。框架层面可以使用统一接口，不必关心底层实现。
