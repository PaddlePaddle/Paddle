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
    所有实现都要与refer的code对比，需要满足精度要求
- 性能测试

# 如何添加新的算子
TBD
## Use me
Add USE_JIT_KERNEL(yourname) to CMakefile.
