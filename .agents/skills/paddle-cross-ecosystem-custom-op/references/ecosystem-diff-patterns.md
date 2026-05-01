# 生态库差异模式

下面不是“哪些文件被改了”的清单，而是“遇到哪类生态库，该先怀疑哪里、先改哪里、怎么避免把 patch 面越做越大”的经验总结。参考仓库主要来自以下对照：

- PFCCLab/DeepEP 对照 deepseek-ai/DeepEP
- PFCCLab/tilelang-paddle 对照 tile-ai/tilelang
- PFCCLab/paddlecodec 对照 meta-pytorch/torchcodec
- PFCCLab/flashinfer 对照 flashinfer-ai/flashinfer
- PFCCLab/FlashMLA 对照 deepseek-ai/FlashMLA
- PFCCLab/sonic-moe 对照 Dao-AILab/sonic-moe
- PFCCLab/DeepGEMM 对照 deepseek-ai/DeepGEMM

实际做 diff 前先确认 `parent` 和默认分支。PFCCLab 这些适配仓库通常把迁移分支设为 `paddle`，而上游多为 `main`；不要直接假设 `main...main`，否则很容易得到空 diff 或错 diff。

## 先记住一个总原则

看 diff 时，不要被“改了很多文件”吓住。真正要提炼的是：

- 这些改动主要集中在哪一层
- 原始问题是 build、注册、Python glue、device/stream/distributed，还是 kernel 本身
- 如果今天要迁移一个新库，第一刀该落在哪里

大多数情况下，答案都不是“先去改 kernel”。

## DeepEP：分布式和通信语义比算子本体更关键

这类库的核心难点通常不在 collective kernel 本身，而在它默认把 PyTorch 的 ProcessGroup、communicator、stream、event 当成现成前提。

### 遇到类似问题时怎么处理

- 先找 runtime glue 层，看 group、rank、world size、stream、event 是在哪里被拿出来的。
- 优先把 communicator/context 的接入点桥接到 Paddle 的 distributed / phi 上下文。
- 不要为了适配分布式，把底层 kernel 或算法路径整体改写。

### 一个具体例子

如果上游代码默认“拿到一个 PyTorch group 对象后，后面所有通信都建立在它的语义之上”，那迁移时应该先把“group 到 communicator/context 的桥”接好，再跑最小单测；而不是把后面每个 collective 调用都换成 Paddle 版本。

### 先看哪些文件

- `setup.py`：先确认 build 入口有没有只在分布式能力上补最小编译标志。
- `csrc/deep_ep.hpp`：看 runtime context、communicator、stream 这些关键成员是怎么被接进来的。
- `csrc/deep_ep.cpp`：看真正的 communicator/context 初始化落点，而不是盯着 collective kernel 本身。
- `deep_ep/buffer.py`：看 Python 侧 group、event、stream 是如何传到底层的。
- `tests/utils.py`：看最小分布式测试是怎样把环境和 group 起起来的。

### 这类库给出的经验

- 先怀疑 distributed glue，不要先怀疑 kernel。
- 只要看到 stream / event / communicator 初始化逻辑，就把它当成一等公民来分析。
- 如果必须依赖 Paddle 分布式内部接口，尽量把依赖收敛在最小入口处，并准备 issue / TODO。

## tilelang-paddle：DSL 生态先修 adapter，不要先碰 lowering

这类库的主体是编译器或 DSL，真正跨框架时最常爆炸的是 current device、current stream、DLPack、JIT runtime 初始化，而不是 DSL 语法本身。

### 遇到类似问题时怎么处理

- 先看 runtime adapter、device helper、stream helper。
- 如果导入时就失败，先查是不是框架初始化 CUDA runtime 的方式不同。
- 如果张量能进来但 kernel 跑不起来，先查 DLPack 和 runtime wrapper，不要立刻改 compiler pass。

### 一个具体例子

如果某个 DSL backend 默认假设“导入库时 PyTorch 已经把 CUDA runtime 准备好了”，那迁移时更可能需要补一个 runtime preload 或 adapter，而不是去改 DSL 生成出来的 kernel。

### 先看哪些文件

- `tilelang/__init__.py`：看导入阶段有没有 runtime preload 或环境准备。
- `tilelang/jit/adapter/base.py`：看 current device/current stream 这类共用入口。
- `tilelang/jit/adapter/tvm_ffi.py`：看具体 backend 是如何把框架张量接入 FFI 的。
- `tilelang/contrib/dlpack.py`：看张量跨框架传递到底依赖哪一层协议。
- `tests_paddle/`：看当前真正被验证过的是哪条 backend 路径，而不是假设所有 backend 都已经通了。

### 这类库给出的经验

- DSL 生态的第一批 patch 应落在 adapter 层。
- DLPack、device、stream 是高频控制点。
- 后端很多时，不要假设所有 backend 都能一次迁过来，先把最常用路径跑通。

## paddlecodec：Python glue 很厚时，先做薄 shim

这类库往往表面上是 C++ custom op，实际上真正复杂的是 Python 层对 `torch.ops`、`torch._dynamo`、buffer 创建、metadata 管理的依赖。

### 遇到类似问题时怎么处理

- 先判断差异是不是还停留在 Python wrapper 层。
- 如果库只是依赖少量 PyTorch 私有行为，优先做最薄的 shim，让调用继续往下走。
- 只有在 Python glue 已经对齐后，才继续追 C++ 或注册层。

### 一个具体例子

如果上游库只是要求“这里存在一个类似 `torch._dynamo` 的对象用于禁用某段图优化”，那迁移时应优先做一个最小 shim 保证语义边界清楚；而不是因为这一个点，把整套 Python 封装重写成 Paddle 原生实现。

### 先看哪些文件

- `src/torchcodec/_core/ops.py`：这是最厚的 Python glue，先看 shim 和 wrapper 有没有先把语义改歪。
- `src/torchcodec/_core/CMakeLists.txt`：看 C++ 扩展最终是怎么链接到 Paddle 侧库的。
- `setup.py`：看打包入口是不是只做了最小切换，而没有破坏原项目布局。
- `src/torchcodec/__init__.py`：看对外 API 形状有没有尽量保持不变。
- `test_paddle/`：看 Paddle 路径当前验证的是哪些最小用户场景。

### 这类库给出的经验

- Python 私有 glue 依赖重时，优先薄 shim。
- shim 本身应被视为临时方案，不要默认长期合理。
- 一旦 shim 超过少量点状补丁，就要反查这是不是 compat gap。

## flashinfer：高性能推理库优先看 runtime feature gate 和框架 bug 边界

这类库通常 kernel 本身高度稳定，真正的跨框架问题更多出在 device 语义、custom op registration、通信路径和 runtime feature gate。

### 遇到类似问题时怎么处理

- 先看库里有哪些“只在某些设备、某些精度、某些通信路径才启用”的逻辑。
- 如果遇到 Paddle 当前行为不兼容，先把 workaround 限定在最小创建路径或最小 wrapper 层。
- 只要发现 workaround 已经在多个地方重复扩散，就停下来准备 issue MRE。

### 一个具体例子

如果某条路径只是在创建张量时触发 Paddle 当前 bug，正确做法是把绕过逻辑限制在这个创建点，并把最小复现抽出来；错误做法是把整个模块里所有张量创建代码都改成另一种写法。

### 先看哪些文件

- `flashinfer/utils.py`：先看 feature gate、device 判断、registration fallback 这些总控逻辑。
- `flashinfer/fused_moe/core.py`：看高频运行路径里是否已经出现框架分支。
- `flashinfer/comm/trtllm_ar.py`：看通信和张量创建路径里有没有被迫加 workaround。
- `flashinfer/decode.py`：看常见用户入口如何把 device/place 语义一路传下去。
- `tests/conftest.py`：看测试入口是不是已经把 compat 范围和环境准备收敛好了。

### 这类库给出的经验

- 高性能库常见问题在 runtime glue，不在 kernel 本体。
- feature gate 是重要线索，很多分支只影响特定路径。
- 看到明确的框架 workaround，就该同步准备 issue，而不是只修当前仓库。

## FlashMLA：主实现基本能跑时，优先隔离测试和 benchmark 差异

这类库常见情况是主算子路径已经大体能被 compat 接住，主要不一致反而集中在测试、benchmark、profiler 和验证脚本。

### 遇到类似问题时怎么处理

- 先判断不一致是不是只发生在测试与 benchmark 辅助代码里。
- 如果主实现不需要动，优先维护独立的 Paddle 测试或验证目录。
- 不要为了让 benchmark 跑起来，把主实现里塞进一堆与业务无关的兼容分支。

### 一个具体例子

如果只有 profiler 路径依赖 PyTorch 的私有上下文管理，而算子本身能正常计算，那么更好的做法是单独适配 benchmark/test harness，而不是把主 `flash_mla_interface.py` 改成满是 profiling 特判。

### 先看哪些文件

- `setup.py`：先确认主构建入口是否几乎没动，只补了 compat 前置准备。
- `paddle_test/`：先区分问题是在主实现，还是在 Paddle 专用验证层。
- `paddle_test/kernelkit/bench.py`：看 benchmark/profiler 适配是否污染了主逻辑。
- `flash_mla/flash_mla_interface.py`：看主算子入口本身是否其实已经足够稳定。

### 这类库给出的经验

- 主实现和测试/benchmark harness 要分开看。
- 如果主路径已通，优先把外围验证体系隔离出来。
- profiler 相关差异往往不能证明算子本体有问题。

## sonic-moe：Triton 与 import-time 假设不兼容时，先包运行时边界

这类库的问题往往不是“数学逻辑不对”，而是 Triton、import 阶段、runtime wrapper 默认把 `torch` 当成真实 PyTorch 模块使用。

### 遇到类似问题时怎么处理

- 先找 import-time patch、runtime wrapper、stream 包装，而不是直接改 Triton kernel。
- 如果必须 monkey patch，也要把 patch 收敛在 import/runtime 边界，不要散到所有调用点。
- 一旦 patch 开始影响大量公共行为，就说明需要重新审视 compat gap。

### 一个具体例子

如果某个 Triton helper 只是要求在编译阶段能看到一组熟悉的 `torch` 命名空间语义，优先在 wrapper 层做最小隔离；不要为了绕开它，把全部 Triton 调用改写成 Paddle 特定版本。

### 先看哪些文件

- `sonicmoe/__init__.py`：先看 import-time patch 和 monkey patch 是否已经集中在入口。
- `sonicmoe/triton_utils.py`：看 Triton 运行时隔离到底包在哪一层。
- `sonicmoe/utils.py`：看 stream、DLPack、wrapper 这些共用辅助逻辑。
- `sonicmoe/moe.py`：看业务入口如何把 runtime 假设串起来。
- `sonicmoe/functional/`：看真正执行前后向路径时，patch 是否已经扩散过深。

### 这类库给出的经验

- import-time patch 越重，越要警惕这是不是 compat 层缺口。
- Triton 生态优先修 wrapper，不优先修 kernel。
- monkey patch 是信号，不是最终方案。

## DeepGEMM：框架无关程度越高，越应该把 patch 收在 build 和 runtime header

这类库的 kernel 和算法几乎不依赖框架，所以迁移时最值钱的原则是：不要因为 build/header 适配而去动真正的 GEMM 逻辑。

### 遇到类似问题时怎么处理

- 先看 `setup.py`、编译标志、JIT runtime header。
- 优先判断是不是宏、include、环境、device runtime 初始化差异。
- 如果主体算法已经足够框架无关，就把改动收敛在 build 和 header 边界。

### 一个具体例子

如果上游只是在编译入口和少量 runtime header 里假设了 PyTorch 扩展环境，那迁移时应该只替换 build 入口和必要宏；不应该因为“顺手”去改 FP8 GEMM 的主体实现。

### 先看哪些文件

- `setup.py`：先看 build 入口是否已经尽量保持最小切换。
- `csrc/jit/device_runtime.hpp`：看 device runtime、流、环境假设是否仍然写死在 PyTorch 语义上。
- `csrc/jit/compiler.hpp`：看 JIT 编译边界需要哪些宏和运行时前提。
- `csrc/python_api.cpp`：看 Python 到 C++ 的入口是否仍然保持原始形状。
- `tests/`：看最小验证是不是已经覆盖真实用户会走的路径。

### 这类库给出的经验

- 框架无关核心越多，越要克制 patch 面。
- 先改 build / header，再动其他层。
- 某些 capability 如果只是被静态假定为存在，要单独验证，不要盲信宏定义。

## 如何把这些案例用于新库迁移

### 先分类，再选参照

你面对的新库，大概率会更像下面四类中的一种：

- 普通 extension：先看 build、compat 头、最小测试
- distributed / stream glue 很重：先看运行时上下文和通信桥接
- Python glue 很重：先看 wrapper、私有 API 依赖、thin shim
- DSL / compiler：先看 adapter、DLPack、current device/current stream

### 套路不是抄代码，而是复用判断顺序

正确复用方式是：

1. 用这些案例判断第一刀该落在哪层。
2. 用对应案例确定“哪些文件通常不该先动”。
3. 只在相似的问题类型上借鉴处理思路。

不要做的事：

- 不要把某个仓库里的 workaround 直接复制到另一个仓库。
- 不要看到 monkey patch 就默认自己也应该 monkey patch。
- 不要因为一个库最后改了很多 Python 文件，就误以为所有新库都应从 Python 层大改开始。

### 一个最终判断标准

如果你的迁移方案开始让上游很难 rebase，或者开始系统性改写上游 API 形状，说明你很可能已经偏离了这些案例真正想传达的模式。
