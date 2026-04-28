# AP DCU Backend 编译与测试指南

## 环境信息

| 项目 | 值 |
|------|------|
| DCU | K100 AI (gfx928 架构, 8 卡) |
| DTK | 25.04.2 |
| Python | 3.10.14 |
| CMake | 3.27.7 |
| GCC | 8.2 |
| HIPCC | /opt/dtk-25.04.2/bin/hipcc |
| Paddle 版本 | 3.5.0.dev20260421 |

---

## 一、编译

### 1.1 环境准备

```bash
# 配置 ldconfig（libgalaxyhip.so.5 等运行时库）
echo /opt/dtk-25.04.2/lib >> /etc/ld.so.conf.d/dtk.conf
ldconfig

# 确认 hipcc 可用
/opt/dtk-25.04.2/bin/hipcc --version
```

### 1.2 编译命令

```bash
cd /work/Paddle/build

# 如果是首次编译，使用以下命令配置：
cmake .. -GNinja -DPY_VERSION=3.10 -DWITH_ROCM=ON -DWITH_CINN=ON -DWITH_DISTRIBUTE=ON

# 编译
ninja -j16
```

### 1.3 编译耗时

整体编译约 1-2 小时（16 线程），主要耗时在 HIP kernel 编译（phi_gpu, phi_core 等）。

### 1.4 编译产物

```
/work/Paddle/build/python/dist/paddlepaddle_dcu-3.5.0.dev20260421-cp310-cp310-linux_x86_64.whl
```

---

## 二、安装

```bash
# 安装 wheel 包
pip install /work/Paddle/build/python/dist/paddlepaddle_dcu-3.5.0.dev20260421-cp310-cp310-linux_x86_64.whl --force-reinstall

# 创建 hytlass 软链接（wheel 包中不包含符号链接，仅 hytlass 核心库需要手动链接）
ln -sf /work/hytlass /opt/py310/lib/python3.10/site-packages/paddle/apy/matmul_pass/matmul/hytlass

# 清除 axpr JSON 缓存（修改 Python 文件后必须执行）
find /opt/py310/lib/python3.10/site-packages/paddle/apy/ -name "*.json" -delete
```

注意：`hytlass_patch/` 和 `hytlass_matmul.h` 已随 PR 提交到 Paddle 仓库中，位于 `python/paddle/apy/matmul_pass/matmul/` 目录下，无需额外操作。只有 `hytlass` 库的核心头文件（`hytlass/include/hytlass/` 和 `hytlass/tools/`）需要通过软链接引入。

---

## 三、测试

### 3.1 环境变量

```bash
export LD_LIBRARY_PATH=/opt/dtk-25.04.2/lib64:/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk-25.04.2/dcc/comgr/lib64:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
```

### 3.2 运行测试

```bash
cd /work/Paddle

# 方式一：pytest
python -m pytest test/ap/test_matmul_add_relu.py -v

# 方式二：直接运行
python test/ap/test_matmul_add_relu.py
```

### 3.3 测试日志

测试日志已保存至 `python/paddle/apy/device/dcu/logs/test_matmul_add_relu.log`

```
============================= test session starts ==============================
platform linux -- Python 3.10.14, pytest-8.3.2, pluggy-1.5.0 -- /opt/py310/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default' -> DirectoryBasedExampleDatabase(PosixPath('/work/Paddle/.hypothesis/examples'))
rootdir: /work/Paddle
configfile: pyproject.toml
plugins: anyio-4.13.0, hypothesis-6.111.2, xdoctest-1.1.1
collecting ... collected 1 item

test/ap/test_matmul_add_relu.py::TestMatmulEpilogue::test_subgraph PASSED [100%]

========================= 1 passed, 1 warning in 9.20s =========================
```

---

## 四、编译问题排查

| 错误信息 | 原因 | 修复方法 |
|----------|------|----------|
| `hiprand/hiprand.h: No such file or directory` | `include_directories(${ROCM_PATH}/include)` 在 `add_subdirectory(paddle)` 之后才生效 | 在 CMakeLists.txt 的 `add_subdirectory(paddle)` 之前添加 `include_directories(${ROCM_PATH}/include)` |
| `cuda.h: No such file or directory` | cupti.cmake 中 CUDA include 路径不存在 | 修改 `${ROCM_PATH}/cuda/include` 为 `${ROCM_PATH}/cuda/cuda/include` |
| `libgalaxyhip.so.5: cannot open shared object file` | DTK 运行时库不在 ldconfig 搜索路径中 | `echo /opt/dtk-25.04.2/lib >> /etc/ld.so.conf.d/dtk.conf && ldconfig` |
| `omp.h: No such file or directory` | hipcc 编译时不应包含 omp.h | 已在代码中用 `!defined(__HIPCC__)` 保护，确保使用最新代码 |
| `std::min 模板参数类型不匹配` | hipcc (clang) 比 nvcc 更严格 | 添加显式类型转换 `static_cast<int>()` |

---

## 五、注意事项

1. **hytlass 软链接**: wheel 包不包含符号链接，安装后必须手动创建。仅需链接 hytlass 核心库，`hytlass_patch/` 和 `hytlass_matmul.h` 已在 Paddle 仓库内
2. **Include 顺序**: `compile_command_util.py` 中 `-I matmul` 在 `-I hytlass/include` 之前，确保优先使用 Paddle 仓库内的 `hytlass_patch/` 和 `hytlass_matmul.h`
3. **JSON 缓存**: 修改 Python 文件后需删除安装目录下的 `.json` 缓存，否则 AP 使用旧序列化代码
4. **HIP_VISIBLE_DEVICES**: 测试时需设置，否则可能检测不到 DCU 设备
5. **LD_LIBRARY_PATH**: 运行时必须包含 DTK 的 lib 目录，否则 DCU 设备无法检测
6. **hipcc 路径**: `compile_command_util.py` 中硬编码了 `/opt/dtk-25.04.2/bin/hipcc`，换 DTK 版本需调整
7. **架构适配**: gfx928 对应 K100 AI，其他 DCU 型号需调整 `--offload-arch` 和 hytlass 中的 arch 标识
