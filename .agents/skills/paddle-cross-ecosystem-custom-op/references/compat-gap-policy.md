# compat 缺口处理

这是本 skill 最重要的约束之一：**遇到 compat gap 时，不要用大范围改写去掩盖问题。**

## 先判断问题属于哪类

### A. Paddle compat 缺口

典型特征：

- 某个常见 `at::*` / `torch::*` / `c10::*` API 根本没被 compat 实现。
- `TORCH_LIBRARY` / `torch.ops` / proxy 行为与已知 compat 测试不一致。
- 生态库依赖的是 PyTorch 公共 API，但在 Paddle compat 下失败。

### B. 生态库依赖了 PyTorch 私有行为

典型特征：

- 依赖 `torch._dynamo`、`torch.profiler`、`torch.library`、内部状态缓存、私有 module side effect。
- 依赖 PyTorch 当前 import 顺序、模块级初始化、副作用或内部 handle。

这类问题不一定是 Paddle bug，但仍然要最小化 workaround，并记录边界。

## 处理顺序

1. 先拿到最小报错点。
2. 把失败收缩成最小复现。
3. 明确这是 compat 缺口还是上游私有假设。
4. 只写最小 workaround，且必须注明删除条件。
5. 如果问题应由 Paddle 修复，准备 issue。

## 最小复现要求

最小复现必须尽量满足：

- 单文件或极小目录结构。
- 最少依赖。
- 明确版本：Paddle commit / wheel 版本、Python、CUDA、驱动。
- 明确命令：build 命令、运行命令。
- 明确期望行为和实际报错。

优先级更高的 MRE 形式：

- 单个 `.py` 脚本
- 极小 `setup.py + csrc/*.cc` 样例
- 如果必须用分布式，再额外给单卡或伪最小脚本说明为什么不能再缩

## Paddle issue 建议模板

标题建议：

```text
[Cross-Ecosystem Custom Op] <具体 API / 行为> is missing or inconsistent in Paddle compat layer
```

正文建议至少包含：

- Paddle 版本 / commit
- Python / CUDA / 驱动版本
- 最小复现代码
- 运行命令
- 期望行为
- 实际行为
- 对照：相同代码在 PyTorch 下是否正常
- 临时 workaround（如果有）

## workaround 允许范围

允许的 workaround：

- 只包住一个具体 incompatibility 点
- 只影响当前库的局部路径
- 不改变公共 API 语义超过必要边界
- 代码里有 TODO，最好有 issue 编号或待跟踪说明

不允许的 workaround：

- 宽泛 `try/except/pass`
- 把大量 `torch` 代码手工翻成 `paddle`，从而失去上游可同步性
- 为绕过一个缺口而顺带重构 unrelated 代码
- 没有说明边界和删除条件的 monkey patch

## TODO 写法建议

推荐写法：

```text
TODO(<owner or issue>): remove this workaround after Paddle compat supports <specific API/behavior>
```

要说清：

- workaround 在解决什么
- 为什么只能先这么做
- 未来怎样删除

## 什么时候必须停下来先报问题

出现以下情况时，不要继续堆 patch：

- 需要连续修改多个核心 kernel 才能绕过 compat 问题
- workaround 已经开始改变库的原始语义
- 需要依赖 Paddle 内部私有 API 才能继续
- 相同模式在多个文件重复出现，说明不是单点问题

这时应该先：

1. 产出最小复现
2. 明确问题边界
3. 在结果里显式标注“当前为 compat gap，不建议继续扩大 patch 面”

## 一条经验法则

如果你发现自己已经开始“系统性地把整个 PyTorch 生态库翻译成 Paddle 原生实现”，说明你很可能已经偏离了这个兼容方案的目标。

正确方向应该是：

- 先保留上游形状
- 让 compat 层尽量承担兼容职责
- 只在 compat 还没覆盖到的局部点做桥接
