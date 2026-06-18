# compat 缺口处理

compat gap 处理的核心目标：把问题边界讲清楚，把 workaround 收敛在最小范围，并为后续的 Paddle 修复准备最小复现。

## 先做问题分类

### A. Paddle compat 覆盖缺口

典型特征：

- 常见 `at::*` / `torch::*` / `c10::*` API 当前没有 compat 实现
- `TORCH_LIBRARY` / `torch.ops` / proxy 行为与现有 compat 测试不一致
- 生态库依赖的是 PyTorch 公共 API，但在 Paddle compat 下失败

### B. 上游仓库依赖 PyTorch 私有行为

典型特征：

- 依赖 `torch._dynamo`、`torch.profiler`、`torch.library`、内部状态缓存、私有 module side effect
- 依赖 PyTorch 当前的 import 顺序、模块级初始化、副作用或内部 handle

这类问题的处理重点是边界说明和最小 shim；是否属于 Paddle bug 要根据最小复现来判断。

## 处理顺序

1. 先拿到最小报错点。
2. 把失败收缩成最小复现。
3. 明确归类为 compat 覆盖缺口还是上游私有假设。
4. 只写最小 workaround，同时写清删除条件。
5. 应由 Paddle 修复的问题，准备 issue；如果只是当前仓库的构建入口选择或上游私有假设，不要制造假的 Paddle issue。

## 最小复现要求

最小复现应尽量满足：

- 单文件或极小目录结构
- 最少依赖
- 明确版本：Paddle commit / wheel 版本、Python、CUDA、驱动
- 明确命令：build 命令、运行命令
- 明确期望行为和实际报错

优先级更高的形式：

- 单个 `.py` 脚本
- 极小 `setup.py + csrc/*.cc` 样例
- 如果必须用分布式，再补一份单卡或伪最小脚本，并说明收缩边界

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

## workaround 边界

workaround 适合在以下条件下使用：

- 只包住一个具体 incompatibility 点
- 只影响当前库的局部路径
- 公共 API 语义保持不变
- 代码里带 TODO，最好有 issue 编号或待跟踪说明

不是所有 Paddle-specific 改动都需要 issue。比如直接使用 `paddle.utils.cpp_extension` 可能只是当前构建系统下最小的入口选择；只有当它明确绕过了 `torch.utils.cpp_extension` proxy / compat 的公共缺口时，才把它升级为 Paddle issue。

一旦出现以下信号，就应该转向 issue 与边界收缩：

- 为一个缺口连续改动多个核心 kernel
- 已经开始改变库的原始语义
- 已经依赖 Paddle 内部私有 API 才能继续
- 相同模式在多个文件重复出现，说明问题已超出单点

## TODO 写法建议

推荐写法：

```text
TODO(<owner or issue>): remove this workaround after Paddle compat supports <specific API/behavior>
```

TODO 至少要说明：

- workaround 在解决什么问题
- 当前为什么需要它
- 未来怎样删除

## 什么时候应立即转向 issue

出现以下情况时，应优先整理 issue MRE：

- 同一调用点上，PyTorch 与 Paddle 的行为已经明确分叉
- 问题来自 compat 公共行为
- workaround 开始在多个文件扩散
- 后续推进已经需要依赖 Paddle 内部私有接口

## 一个总判断标准

如果当前方案已经开始系统性改写整个 PyTorch 生态库的 API 形状，说明补丁边界需要回收。兼容方案应尽量保留上游形状，让 compat 层承担兼容职责，只在缺口位置放置最小桥接。
