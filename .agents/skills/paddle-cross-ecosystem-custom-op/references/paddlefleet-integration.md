# PaddleFleet 集成

生态库完成 Paddle 迁移后，一部分会集成进 PaddleFleet，对外统一通过 `paddlefleet_ops` 暴露。这份文档回答三个问题：集成机制长什么样、集成一个新库的标准步骤、eager import 硬约束怎么满足。

## 集成机制总述

- **进入方式**：生态库以 git submodule 挂在 `packages/paddlefleet_ops/third_party/<lib>`，指向 PFCCLab 的 Paddle 适配 fork 并固定到具体 commit。不是 vendored 拷贝，也不是 pip 依赖。
- **构建**：`packages/paddlefleet_ops/build_utils.py` 的 `get_libs()` 返回 `EcosystemLibrary` 列表。构建 wheel 时每个库被独立 `pip install --target <临时目录> --no-deps --no-build-isolation`，再按 `Artifact(source, target)` 把产物收进 `src/paddlefleet_ops/<name>`，一起打进 `paddlefleet_ops` wheel。产物目录要加进 `.gitignore`。
- **加载**：`paddlefleet_ops/__init__.py` 在模块顶层（即首次 `import paddlefleet_ops` 时）逐库执行：用 `with paddle.use_compat_guard(scope={...}, silent=True):` 包住 `_safe_load_ecosystem_lib("<lib>", ops_dir, globals())`，以顶层名导入并把 `sys.modules` 挂到 `paddlefleet_ops.<lib>` 命名空间下，guard 退出后 compat 状态自动恢复。没有任何按需/延迟加载路径。
- **硬件/环境门槛不满足时**：不加载，错误信息记入 `blocked_import_messages`，由 meta path blocker 让后续 `import paddlefleet_ops.<lib>` 抛出带指引的 `RuntimeError`；对外提供 `is_<lib>_available()` 供上层守卫。

运行时注册的代码形状（新集成一律用 `use_compat_guard`；早期集成用的 `paddle.enable_compat(scope=...)` 已在仓库里逐步替换，不要再新增这种写法）：

```python
if is_sonic_moe_available():
    with paddle.use_compat_guard(
        scope={"sonicmoe", "quack", "triton"}, silent=True
    ):
        _safe_load_ecosystem_lib("sonicmoe", ops_dir, globals())
else:
    blocked_import_messages["paddlefleet_ops.sonicmoe"] = error
```

`use_compat_guard` 是 context manager（也可以当装饰器用，行为以 `python/paddle/compat/proxy.py` 实现为准），只在 `with` 块内启用 proxy、退出即恢复原状态；配合 eager import 约束（库在 import 阶段加载完全部子模块），guard 结束后不需要残留任何全局 compat 状态。scope 要把库自身及其会在加载期 `import torch` 的依赖都列上（如上例的 `quack`、`triton`）。

构建注册的代码形状（CUDA extension 型库，用 `extra_env` 控制目标架构）：

```python
EcosystemLibrary(
    name="FlashMLA",
    source_rel_path="third_party/FlashMLA",
    artifacts=[Artifact("flash_mla", "flash_mla")],
    extra_env={"PADDLE_CUDA_ARCH_LIST": ""},
)
```

## 直接参考的模板 PR

集成方式已经比较固定，按库的类型对照相应 PR 即可：

| PR | 库 | 类型 | 集成上的分叉点 |
|---|---|---|---|
| [PaddleFleet#1114](https://github.com/PaddlePaddle/PaddleFleet/pull/1114) | cudnn-frontend | header-only C++ + pybind11 | 编译依赖（ninja、pybind11）要同时加进 `pyproject.toml` build requires 和 CI workflow；门槛是 Python 版本而非 GPU 架构 |
| [PaddleFleet#1132](https://github.com/PaddlePaddle/PaddleFleet/pull/1132) | FlashMLA | CUDA extension | `extra_env` 控制目标架构；门槛是 compute capability；配一个与现有实现对比精度的单测 |
| [PaddleFleet#1589](https://github.com/PaddlePaddle/PaddleFleet/pull/1589) | MoonEP | EP 通信库 | arch 列表复用 DeepEP；运行时 pip 依赖走 `setup.py` 的 `get_special_setup_deps()`；验证必须是多卡测试并注册进 `tests/test_configs.yaml` |

一个例外：tilelang-paddle 不走 submodule，而是作为 pip 依赖声明在 PaddleFleet 根 `pyproject.toml` 的主包 dependencies 里，顶层 import 名仍是 `tilelang`，compat 由使用方自行开启。除非有同样明确的理由，新库默认按模板 PR 的方式集成。

## 集成新库的标准步骤

1. `.gitmodules` 加 submodule，路径 `packages/paddlefleet_ops/third_party/<Lib>`，URL 指向 `https://github.com/PFCCLab/<Lib>.git`，固定 commit。
2. `build_utils.py` 的 `get_libs()` 追加 `EcosystemLibrary(...)`，按需设 `extra_env` / `include_dirs`；有额外门槛（CUDA / Python 版本）就放条件分支；同时把库名加进 `check_submodule_updated()` 名单。
3. 声明依赖：编译期依赖加 `packages/paddlefleet_ops/pyproject.toml` 的 `[build-system] requires`；运行时 pip 依赖加 `setup.py` 的 `get_special_setup_deps()`。
4. `.gitignore` 加 `packages/paddlefleet_ops/src/paddlefleet_ops/<目标名>`。
5. `paddlefleet_ops/__init__.py` 按既有模板加载：可用性标志与 `is_<lib>_available()`、提示文案、顶层 `with paddle.use_compat_guard(scope={...}, silent=True):` 包住 `_safe_load_ecosystem_lib`、else 分支填 `blocked_import_messages`。
6. 检查 eager import 前置条件（见下节）；不满足先改生态库 fork，不要改集成方式。
7. 若引入新编译依赖，同步更新构建 ops wheel 的 CI workflow 里的 pip install 行。
8. 加测试：`tests/single_card_tests/custom_ops/test_ops_import.py` 加 import 冒烟测试；功能单测按单卡/多卡放置，多卡用例注册进 `tests/test_configs.yaml`；统一用 `is_<lib>_available()` 做 skip 守卫。
9. （可选）在 `src/paddlefleet/` 加上层包装，import 用 `try/except (ImportError, RuntimeError)` 兜底。
10. 本地验证：`git submodule update --init --recursive` → 构建 `paddlefleet-ops` wheel → 用 wheel 跑 import 测试。

## compat guard 内的 eager import 约束

首次 `import paddlefleet_ops` 时，要在加载生态库的 `use_compat_guard` 块内完成所有**依赖 compat proxy 的模块初始化**。这里要求 eager 的是会执行 `import torch`、解析 proxy 类型或依赖其他 compat 状态的模块，不是无条件加载包内每个可选依赖。

原因：compat proxy 只在对应的 `use_compat_guard` 块内生效。依赖 proxy 的模块如果在 guard 退出后才首次加载，它们的 `import torch` 和相关类型解析就不再处于集成时验证过的 compat 上下文中。

### 需要修掉的两类 proxy-sensitive lazy import

第一类：`__init__.py` 没有 eager import 子模块。

```python
# custom_op/__init__.py
# （没有 import submodule）

# custom_op/submodule.py
import torch
```

`import custom_op` 阶段加载不到 `custom_op.submodule`，其中的 `import torch` 就不会被 proxy。修复方式是在 `__init__.py` 里补上 eager import：

```python
# custom_op/__init__.py
from . import submodule
```

第二类：`import torch`，或对依赖 compat proxy 的子模块的 import，写在函数体内。

```python
# custom_op/submodule.py
def fn():
    import torch

    ...
```

这个 import 要等 `fn` 被调用才执行，届时可能已出 proxy 生效范围。修复方式是提到模块顶层：

```python
# custom_op/submodule.py
import torch


def fn(): ...
```

一个真实例子：flash-linear-attention fork 的提交 [`683b34f3`](https://github.com/PFCCLab/flash-linear-attention/commit/683b34f3) 把上游原注释为 "Import here to avoid circular dependency" 的函数体内延迟 import 全部提升到模块顶层，并把以 lazy import 著称的 `fla/__init__.py` 改成顶层 `from fla import modules, ops`。

### 可以保留 lazy import 的边界

如果某个依赖在模块初始化阶段会因为 AST 预处理、类型反射等原因与 compat proxy 冲突，可以保留局部 lazy import。比如 [quack](ecosystem-cases/quack.md) 的 `cutlass.torch` 在模块顶层加载时会反射 compat 下的 `torch.device`，因此需要把 import 隔离到实际使用它的函数路径。

这类例外要同时满足以下条件：

- import 收敛在实际使用该依赖的最小函数或分支中，普通包初始化路径不会触发它。
- 明确该调用路径是否仍依赖 compat proxy；如果依赖，就在调用点建立对应的 compat 上下文或在生态库 fork 中做显式桥接，不能假定首次 import 时的 guard 仍然有效。
- 临时 proxy、module swap 或 guard 状态不能泄漏到调用 scope 之外；如果缓存导入结果，要确认它在后续调用中的语义仍然成立。
- 在代码注释中写明模块顶层 import 失败的原因、触发路径和删除条件，并分别验证包初始化路径与实际调用路径。

### circular import 的取舍

把 lazy import 改成 eager import 可能引入 circular import。这时需要根据情况对部分模块做取舍调整——例如手动控制顶层导入顺序：

```python
# initialize FLA ops before the convolution backend to avoid the modules/ops import cycle.
# isort: off
from fla.ops.cp import (
    FLACPContext,
    conv_cp_send_recv_bwd,
    conv_cp_send_recv_fwd,
)
from fla.modules.conv.triton.ops import causal_conv1d_bwd, causal_conv1d_fwd
# isort: on
```

优先通过导入顺序或模块边界消除循环依赖，不要用 proxy-sensitive lazy import 掩盖循环。满足上一节兼容性例外时，可以保留局部 lazy import，但要按其作用域和生命周期要求验证。

## 集成问题的处理边界

普通的 proxy-sensitive lazy import 应回到生态库 fork 修正；因 AST 或类型反射必须延迟的依赖，也应在 fork 内完成局部隔离或显式桥接。不要仅为绕过生态库的导入问题修改 PaddleFleet 的统一加载方式。

## 集成前自查清单

- 生态库的迁移分支能独立 build / import / 跑通最小测试。
- 除已按上述边界记录的兼容性例外外，生态库 `__init__.py` 递归地 eager import 了所有依赖 compat proxy 的子模块。
- 库内没有未经说明的函数体内 `import torch` 或 proxy-sensitive 子模块 import。
- 因 AST / 类型反射保留的 lazy import 已注明触发 scope、compat 上下文、状态生命周期和删除条件，并覆盖包初始化与实际调用测试。
- 改动引入的 circular import 已经解决，没有重新引入不符合上述边界的 proxy-sensitive lazy import。
