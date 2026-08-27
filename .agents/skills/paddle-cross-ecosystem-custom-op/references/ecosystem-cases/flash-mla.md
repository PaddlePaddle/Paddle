# FlashMLA：控制面在验证层隔离

这类仓库的常见情况是主算子路径已经基本跑通，主要差异集中在测试、benchmark、profiler 和验证脚本。

### 第一落点

- Paddle 专用验证层
- benchmark / profiler harness

### 具体例子

主构建入口的全部改动只有 4 行——在 import `torch.utils.cpp_extension` 前开启 compat 代理，build 系统与 C++ 主体零改写（这是 compat 机制的设计用法，长期保留）：

```diff
--- setup.py
 from setuptools import setup, find_packages

+import paddle
+
+paddle.enable_compat()
+
 from torch.utils.cpp_extension import (
     BuildExtension,
     CUDAExtension,
```

而 diff 里主实现目录（`csrc/`、`flash_mla/`）零改动，`flash_mla/flash_mla_interface.py` 根本没出现在 diff 中。真正成规模的改动是验证层：上游 benchmark 依赖 torch kineto profiler 的事件流，Paddle 侧没有等价接口，于是在 `paddle_test/` 里新增一套基于 `paddle.profiler` + trace.json 解析的计时 harness（`paddle_test/kernelkit/bench.py` 新增文件），主算子代码不动。如果 compat 未来对齐 `torch.profiler` 事件接口，这套平行 harness 可以收敛回上游路径（部分可吸纳）。

### 优先查看的文件

- `setup.py`：确认主构建入口的修改规模
- `paddle_test/`：区分主实现问题与 Paddle 专用验证问题
- `paddle_test/kernelkit/bench.py`：看 benchmark/profiler 适配如何组织
- `flash_mla/flash_mla_interface.py`：看主算子入口是否已经稳定

### 可复用结论

- 主实现和验证层要分开判断
- 主路径已通时，外围验证体系适合独立收敛
- profiler 差异本身不能直接推导出算子主体有问题

## FlashMLA（paddle-train 变体分支）：控制面只剩 build 入口的一个 compat 开关

`paddle-train` 基于上游 deepseek-ai/FlashMLA 的 `nv_dev` 开发分支（训练所需的 sparse/DSA 内核演进），与面向推理的 `paddle` 主迁移分支 base 不同、按 cherry-pick 共享同一个 compat 提交。对 `nv_dev` 的比较结果是：整条分支的 Paddle 适配只有一个提交、一个文件、4 行——就是 FlashMLA case 里展示过的那段 `setup.py` 里 `paddle.enable_compat()` 前置。

### 第一落点

- `setup.py`：`paddle.enable_compat()` 放在所有 `torch` import 之前，torch extension 构建链随即被代理
- 没有第二落点：C++、Python 包装、测试全部零改动

### 具体例子

训练路径相对推理路径多出来的部分——`flash_mla/flash_mla_interface.py` 里带 `save_for_backward` 和 `backward` 的 `torch.autograd.Function` 子类、`csrc/sm100/prefill/dense/` 下的 cutlass backward 内核——在这条分支上全部是未修改的上游代码，autograd 注册、反向传播、dtype 处理直接经 compat 代理运行。对比 `paddle` 分支还维护了 `paddle_test(s)/` 验证套件，`paddle-train` 连测试都直接复用上游 `tests/`。

### 优先查看的文件

- `setup.py`：整条分支唯一的 Paddle 落点（4 行）
- `flash_mla/flash_mla_interface.py`：确认 `torch.autograd.Function` 训练路径零改动跑在 compat 上
- `csrc/api/api.cpp`：上游 pybind 注册入口的形状（compat headers 直接消化）
- `tests/lib.py`：上游测试基础设施，未做 Paddle 化改造

### 可复用结论

- compat 机制成熟后，变体分支的边际成本可以压到"一个提交、几行 build 开关"；为上游活跃开发分支开对应变体几乎零负担，rebase 就是重放一个提交
- 变体分支该不该独立开，看 base 是否不同；补丁共享不靠分支间 merge，而是把 compat 提交做成可独立 cherry-pick 的最小单元，并随 compat API 演进在各分支同步更新
- autograd/backward 这类"训练特有"的部分不必然带来额外适配面——只要建立在被 compat 覆盖的公共 API 上，就归入"保持不变的部分"
- 主迁移分支上补的专用验证层不是变体分支的必需品；新变体可以先靠上游测试 + compat 跑通，验证层按需复制
