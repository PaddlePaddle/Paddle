# JIT Kernel

结合函数模板和JIT生成需要的kernel函数。
这里的kernel是比Operator中kernel更小级别的算子单元，更侧重的是在不同硬件上的性能。
目前仅支持CPU上的高性能计算。

## 目录结构

```txt
PaddlePaddle/Paddle/paddle/fluid/
├── ...
├── operator/
│   ├── .../
└── jit/
    ├── ...
    ├── gen/
    │   └── ...
    |── more/
    │   ├── ...
    │   ├── mkl/
    │   │   └── ...
    │   └── openblas/
    │       └── ...
    └── refer/
        └── ...
```

基础class都的根目录下，根目录下包括jitcode,more和refer。每个目录下都是一种实现，每种kernel算子都需要有reference的实现，其他的都是可选的。
- jitcode： 代表使用jit生成的code，需要依赖xbyak。他关心的是性能。
- refer：代表reference的实现，每种kernel算子都需要有在CPU上的reference的实现，他主要关心的算法逻辑。
- more： 下面可以放入跟多实现，包括mkl，mkldnn，openblas等，也可以是自身已有的kernel组合。

## 动态获取

提供一个get方法，根据kernel类别获取，每种实现都有自己的使用范围，根据范围动态和当前条件选择需要的kernel函数。

## 测试

- 逻辑测试
    所有实现都要与refer的code对比，需要满足精度要求， 包括float和double的数据类型
- 性能测试
    所有实现的性能对比，并且与最终的`jit::Get`方法对比，该方法拿到的性能需要是最好的。

# 如何添加新的算子

- 在`KernelType` 中添加 `your_key` .
- 实现Reference 的逻辑，每个jitkernel的Reference 实现是必须的。不要依赖任何第三方库。并在`refer/CmakeLists.txt`中`USE_JITKERNEL_REFER(your_key)`.
- (optional) 实现更多的算法在`more`目录下，可以依赖mkl，openblas，或者mkldnn等第三方库。
- (optional) 实现基于Xbyak的生成code，在`gen`目下。 jitcode需要实现自己的`JitCodeCreator`，并注册在KernelType上。
- 必要时可以添加新的`KernelTuples`，可以参考`XYZNTuples`，新加的Attr类型需要特例化`JitCodeKey`方法。
- 添加unit test，需要测试float和double
- 添加benchmark确保get得到的速度是最快。
