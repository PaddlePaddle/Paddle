# AP DCU Backend Fix - 基于 PR #78727 适配 hytlass

## 背景

PR #78727 ([AP] Add ap dcu backend) 为 Paddle AP 添加了 DCU 后端支持，采用的是 **CK (Composable Kernel)** 方案。
DCU 侧需要改为 **hytlass** (cutlass 的 HIP/ROCm 移植版) 方案，并在 DCU 机器上跑通单测。

## 修改文件清单

### 1. python/paddle/apy/matmul_pass/matmul/matmul.h

**改动**: HIPCC 分支从 ck_patch/ck_matmul.h 改为 hytlass_matmul.h，去掉 math_function.h（含 CUDA 专有头文件）

- 修改前: include "ck_patch/ck_matmul.h" + include "math_function.h"
- 修改后: include "hytlass_matmul.h" （去掉 math_function.h）

### 2. python/paddle/apy/matmul_pass/matmul/profile.h

**改动**: HIPCC 分支从 ck_patch/check.h 改为 hytlass_patch/check.h

- 修改前: include "ck_patch/check.h"
- 修改后: include "hytlass_patch/check.h"

### 3. python/paddle/apy/device/dcu/compile_command_util.py

**改动**: 从 CK 编译命令改为 hytlass 编译命令，并调整 Include 顺序

关键差异:
- 编译器: /opt/dtk-25.04.2/bin/hipcc（需要完整路径，因默认 PATH 中无 hipcc）
- 标准: -std=c++17（hytlass 需要 C++17，而非 CK 的 C++20）
- 架构: --offload-arch=gfx928（K100 AI DCU 架构，而非 gfx906）
- Include 路径: hytlass/include + hytlass/tools/util/include（而非 CK 的 composable_kernel/include）
- **Include 顺序**: `-I matmul` 在 `-I hytlass/include` 之前，确保优先使用 Paddle 仓库内的 hytlass_patch/ 和 hytlass_matmul.h

```python
def generate_matmul_compile_command(self, tpl_dirname, library_name):
    hytlass_dir = f"{tpl_dirname}/matmul/hytlass"
    matmul_source_dir = f"{tpl_dirname}/matmul"

    compile_cmd = "/opt/dtk-25.04.2/bin/hipcc -std=c++17 -O3 -fPIC --offload-arch=gfx928 -Wno-return-type"
    compile_cmd = compile_cmd + " -I " + matmul_source_dir        # 优先：Paddle 仓库的 hytlass_patch/ 和 hytlass_matmul.h
    compile_cmd = compile_cmd + " -I " + hytlass_dir + "/include"  # 其次：hytlass 核心库头文件
    compile_cmd = compile_cmd + " -I " + hytlass_dir + "/tools/util/include"
    compile_cmd = (
        compile_cmd + " -DAP_ENABLE_AUTOTUNE=0 -DAP_ENABLE_DEBUG=0"
    )
    compile_cmd = (
        compile_cmd
        + f" --shared {library_name}.{self.file_ext} -o lib{library_name}.so"
    )
    return compile_cmd
```

### 4. python/paddle/apy/matmul_pass/matmul/hytlass_patch/ (新增目录)

**改动**: 将 hytlass_patch 从 hytlass 库复制到 Paddle 仓库

最初 hytlass_patch 仅位于 hytlass 库的 `include/hytlass_patch/` 目录下，需要通过软链接引入。
后来按需求将 hytlass_patch 移入 Paddle 仓库，随 PR 一起提交。

目录结构：
```
hytlass_patch/
├── check.h
├── all_tuning_configs.h
├── batched_matrix_coord.h
├── trace_device.h
├── epilogue/
│   └── thread/
│       ├── linear_combination_unary.h
│       └── linear_combination_variadic.h
└── gemm/
    ├── device/
    │   └── gemm_universal_with_variadic.h
    └── kernel/
        └── default_gemm_with_variadic.h
```

`hytlass_patch`不是直接复用`cutlass_patch`，而是在hytlass库fork cutlass时已将patch文件中的命名做了替换（`cutlass/` → `hytlass/`、`namespace cutlass` → `namespace hytlass`、`CUTLASS_HOST_DEVICE` → `HYTLASS_HOST_DEVICE`）。

### 5. python/paddle/apy/matmul_pass/matmul/hytlass_matmul.h (新增文件)

**改动**: 将 hytlass_matmul.h 从 hytlass 库复制到 Paddle 仓库

最初 hytlass_matmul.h 仅位于 hytlass 库的 `include/hytlass_matmul.h`，需要通过 `-I hytlass/include` 引入。
后来按需求将此文件移入 Paddle 仓库，随 PR 一起提交。

该文件是 hytlass matmul 的入口头文件，包含了 hytlass gemm kernel 的头文件和 `MatmulAddVariadic` 模板。

### 6. python/paddle/apy/matmul_pass/matmul/.apy_ignore

**改动**: composable_kernel 替换为 hytlass

```
cutlass
generate_configs.py
hytlass
```

### 7. test/ap/test_matmul_add_relu.py

**改动**:
- 去掉 `IsCertainDevices()` 设备判断（原来是只检测 A100 才跑数值校验，DCU 和其他设备不校验）
- `pcc.compile()` 的 `backend_device` 参数改为根据编译平台自动选择
- 无条件执行数值正确性校验（`np.testing.assert_allclose`）

```python
# 修改前:
def IsCertainDevices():
    try:
        sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str = sp.communicate()[0].decode('utf-8')
        if 'A100' in out_str:
            return True
        else:
            return False
    except Exception as e:
        return False

fused_foo = pcc.compile(foo, ap_path=...)  # backend_device 默认 'cuda'
if IsCertainDevices():  # 只有 A100 才校验正确性
    ap_outs = fused_foo(self.x, self.y, self.b)
    dy_outs = foo(self.x, self.y, self.b)
    np.testing.assert_allclose(dy_outs, ap_outs, atol=1e-1)

# 修改后:
backend_device = 'dcu' if paddle.is_compiled_with_rocm() else 'cuda'
fused_foo = pcc.compile(foo, ap_path=..., backend_device=backend_device)
# 所有设备都校验正确性
ap_outs = fused_foo(self.x, self.y, self.b)
dy_outs = foo(self.x, self.y, self.b)
np.testing.assert_allclose(dy_outs, ap_outs, atol=1e-1)
```

### 8. CMakeLists.txt (编译修复)

**改动**: 在 add_subdirectory(paddle) 之前添加 ROCM include 路径

```cmake
if(WITH_ROCM)
  include_directories(${ROCM_PATH}/include)
endif()

add_subdirectory(paddle)
```

**原因**: hip.cmake 中的 `include_directories(${ROCM_PATH}/include)` 未正确传播到 cc_library 目标，
导致 `hiprand/hiprand.h` 等头文件找不到。在 add_subdirectory(paddle) 之前显式添加可以确保传播。

### 9. cmake/cupti.cmake (编译修复)

**改动**: 修正 CUDA 兼容头文件路径

```cmake
# 修改前:
include_directories(${ROCM_PATH}/cuda/include)
# 修改后:
include_directories(${ROCM_PATH}/cuda/cuda/include)
```

**原因**: DTK 25.04.2 中 CUDA 兼容头文件位于 `${ROCM_PATH}/cuda/cuda/include/`，
而非 `${ROCM_PATH}/cuda/include/`（该路径不存在），导致 `cuda.h` 找不到。

### 10. phi/kernels/funcs/mode.h, check_numerics_utils.h, viterbi 相关文件 (编译修复)

**改动**: 在 `#include <omp.h>` 的 PADDLE_WITH_MKLML 宏保护中加入 `!defined(__HIPCC__)` 条件

```cpp
// 修改前:
#ifdef PADDLE_WITH_MKLML
#include <omp.h>
#endif

// 修改后:
#if defined(PADDLE_WITH_MKLML) && !defined(__HIPCC__)
#include <omp.h>
#endif
```

**原因**: hipcc (clang) 无法找到 GCC 的 omp.h，且 omp.h 在 GPU kernel 中不需要。

### 11. phi/kernels/gpu/interpolate_kernel.cu, interpolate_grad_kernel.cu (编译修复)

**改动**: 修复 std::min 模板参数类型不匹配

```cpp
// 修改前:
int grid_z = std::min(static_cast<int>(nc), gpu_props.maxGridSize[2]);
// 修改后:
int grid_z = std::min(static_cast<int>(nc), static_cast<int>(gpu_props.maxGridSize[2]));
```

**原因**: maxGridSize 是 uint32_t，static_cast<int>(nc) 是 int，std::min 要求两参数类型相同，
hipcc (clang) 比 nvcc 更严格。

### 12. cmake/hip.cmake (编译修复)

**改动**: 添加 HIP include 路径

```cmake
include_directories(${HIP_PATH}/include)  # 已有行
include_directories(${ROCM_PATH}/include)  # 新增行（原 hip.cmake 中有但未传播）
```

## 编译测试

### 1、编译 Paddle

拉取 Paddle PR：https://github.com/PaddlePaddle/Paddle/pull/78822

使用如下命令编译 Paddle：

```bash
$ cd Paddle
$ echo /opt/dtk-25.04.2/lib >> /etc/ld.so.conf.d/dtk.conf
$ ldconfig
$ mkdir build
$ cd build
# AP依赖CINN，需设置WITH_CINN=ON；DCU需设置WITH_ROCM=ON
$ cmake .. -GNinja -DPY_VERSION=3.10 -DWITH_ROCM=ON -DWITH_CINN=ON -DWITH_DISTRIBUTE=ON
$ ninja -j16
# 安装whl包
$ pip install ./python/dist/paddlepaddle_dcu-3.5.0.dev20260421-cp310-cp310-linux_x86_64.whl
```

### 2、执行测试

将hytlass库拷贝（或软链接）到Paddle安装目录的`paddle/apy/matmul_pass/matmul/hytlass`。

`paddle/apy`是AP的文件目录，目前暂未支持自动打包到whl包中。DCU相关的AP目录结构如下：
```
paddle/apy
├── device
│   ├── cuda
│   │   └── compile_command_util.py
│   └── dcu
│       ├── compile_command_util.py
│       └── logs
│           └── test_matmul_add_relu.log
└── matmul_pass
    ├── matmul
    │   ├── ck_patch          # CK (Composable Kernel) patch，原方案，不再使用
    │   ├── cutlass_patch     # cutlass patch，CUDA侧使用
    │   ├── hytlass_patch     # hytlass patch，DCU侧使用（已随PR提交到Paddle仓库）
    │   ├── hytlass           # hytlass库核心头文件（软链接，需手动创建）
    │   ├── hytlass_matmul.h  # hytlass matmul入口头文件（已随PR提交到Paddle仓库）
    │   ├── matmul.h
    │   ├── params.h
    │   └── profile.h
    ├── matmul_epilogue_pass.py
    ├── matmul_variadic_ptn.py
    └── matmul_variadic_tpl.py
```

其中`hytlass_patch`与`cutlass_patch`同级，位于Paddle仓库内，已随PR一起提交。

`hytlass_matmul.h`也已随PR提交到Paddle仓库，编译时通过`-I matmul`（在`-I hytlass/include`之前）优先从Paddle仓库本地引用。

`hytlass`目录下需要手动创建软链接的关键文件（仅hytlass库核心头文件）：
```
hytlass
├── include
│   └── hytlass/              # hytlass库核心头文件
│       ├── epilogue/
│       ├── gemm/
│       ├── util/
│       └── ...
└── tools
    └── util
        └── include
```

创建hytlass软链接并清除缓存：

```bash
# 创建 hytlass 软链接（wheel 包中不包含符号链接）
$ ln -sf /work/hytlass /opt/py310/lib/python3.10/site-packages/paddle/apy/matmul_pass/matmul/hytlass
# 清除 axpr JSON 缓存
$ find /opt/py310/lib/python3.10/site-packages/paddle/apy/ -name "*.json" -delete
```

测试文件：`test/ap/test_matmul_add_relu.py`，测试子图为matmul + relu + add融合：
```python
def foo(
    x: pct.Tensor([B, M, K], DType),
    w: pct.Tensor([K, N], DType),
    b: pct.Tensor([B, M, N], DType),
):
    y = paddle.matmul(x, w)
    tmp = paddle.nn.functional.relu(y)
    tmp2 = tmp + b
    return tmp2
```

执行测试：

```bash
# 设置环境变量
$ export LD_LIBRARY_PATH=/opt/dtk-25.04.2/lib64:/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk-25.04.2/dcc/comgr/lib64:$LD_LIBRARY_PATH
$ export HIP_VISIBLE_DEVICES=0
$ cd /work/Paddle
$ python -m pytest test/ap/test_matmul_add_relu.py -v
```

完整的执行日志已上传到PR中。通过日志可以看到，成功生成了`pd_op.ap_variadic`算子，并使用hytlass编译了matmul kernel：
```
I0428 15:12:13.623298 35229 add_pcc_pass.cc:134] Compiling subgraph with PCC backend ...
E0428 15:12:13.623658 35229 add_pcc_pass.cc:122] 0) after ApplyApFacadePass():
{
    (%3) = "pd_op.matmul" (%0, %1) ...
    (%4) = "pd_op.relu" (%3) ...
    (%5) = "pd_op.add" (%4, %2) ...
}
...
E0428 15:12:13.889163 35229 add_pcc_pass.cc:122] 2) after ApplyApGenericDrrPass():
{
    (%6) = "builtin.combine" (%0, %1, %2) ...
    (%7) = "pd_op.ap_variadic" (%6) {
      code_module_lambda: ... "/opt/dtk-25.04.2/bin/hipcc -std=c++17 -O3 -fPIC --offload-arch=gfx928 ..."
      infer_meta_lambda: ...
    }
    (%8) = "builtin.split" (%7) ...
}
PASSED

========================= 1 passed, 1 warning in 8.94s =========================
```

## 注意事项

1. **hytlass 软链接**: wheel 包不会包含符号链接，安装后需要手动创建。仅需链接 hytlass 核心库，`hytlass_patch/` 和 `hytlass_matmul.h` 已在 Paddle 仓库内。
2. **Include 顺序**: `compile_command_util.py` 中 `-I matmul` 在 `-I hytlass/include` 之前，确保优先使用 Paddle 仓库内的 `hytlass_patch/` 和 `hytlass_matmul.h`，而非 hytlass 库自带的同名文件。
3. **hipcc 路径**: compile_command_util.py 中使用了 /opt/dtk-25.04.2/bin/hipcc 硬编码路径，不同 DTK 版本可能需要调整。
4. **架构适配**: gfx928 对应 K100 AI DCU，其他 DCU 型号可能需要调整 --offload-arch 和 hytlass_matmul.h 中的 Gfx928 arch 标识。
5. **JSON 缓存**: 修改 Python 文件后需要删除安装目录下的 .json 缓存文件，否则 AP 会使用旧的序列化代码。
6. **LD_LIBRARY_PATH**: 运行时需要包含 `/opt/dtk-25.04.2/lib64` 和 `/opt/dtk-25.04.2/dcc/comgr/lib64`，否则 DCU 设备无法检测。
7. **ldconfig 配置**: 编译和运行时需要将 `/opt/dtk-25.04.2/lib` 加入 ldconfig，否则 wheel 打包步骤会因找不到 `libgalaxyhip.so.5` 而失败。
8. **预存编译问题**: 编译修复中的第4、5项（omp.h 和 std::min）是 DCU 平台上的预存问题，与 AP DCU 后端改动无关，但需要修复才能完成完整编译。
