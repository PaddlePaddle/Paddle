# macOS (Apple Silicon) 源码编译

在 macOS Apple Silicon (M1/M2/M3/M4 等) 上从源码编译 Paddle 的完整流程。仅支持 CPU 模式（无 GPU/CUDA）。

## 前置依赖

通过 Homebrew 安装：

```bash
brew install cmake ninja uv
```

需要的工具和最低版本：
- **cmake** >= 3.19.2（Apple Silicon 支持从此版本开始）
- **ninja**（推荐，比 make 快很多）
- **uv**（Python 环境管理）
- **Xcode Command Line Tools**：`xcode-select --install`

## 编译流程

### 1. 创建 Python 虚拟环境

```bash
cd /path/to/Paddle
PY_VERSION=3.10
VENV_DIR=venvs/paddle-py${PY_VERSION//./}

uv venv --seed ${VENV_DIR} --python=${PY_VERSION}
source ${VENV_DIR}/bin/activate
uv pip install -r python/requirements.txt
```

### 2. 获取 Python 路径

macOS 上需要显式指定 Python 库和头文件路径：

```bash
PY_LIB_DIR=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PY_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PY_LIBRARY=${PY_LIB_DIR}/libpython${PY_VERSION}.dylib
```

### 3. CMake 配置

```bash
mkdir -p build && cd build
rm -rf CMakeCache.txt CMakeFiles/    # 首次或切换配置时需要清理

cmake .. \
    -GNinja \
    -DPY_VERSION=${PY_VERSION} \
    -DPYTHON_INCLUDE_DIR=${PY_INCLUDE_DIR} \
    -DPYTHON_LIBRARY=${PY_LIBRARY} \
    -DWITH_GPU=OFF \
    -DWITH_ARM=ON \
    -DWITH_AVX=OFF \
    -DWITH_TESTING=ON \
    -DWITH_DISTRIBUTE=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

关键选项说明：

| 选项 | 值 | 原因 |
|------|----|------|
| `WITH_GPU` | OFF | macOS 无 CUDA |
| `WITH_ARM` | ON | Apple Silicon 是 ARM 架构 |
| `WITH_AVX` | OFF | ARM 无 AVX 指令集，`WITH_ARM=ON` 时自动禁用 |
| `WITH_DISTRIBUTE` | OFF | macOS 不支持 NCCL 等分布式通信库 |
| `WITH_TESTING` | ON/OFF | ON 编译测试二进制（更慢），OFF 跳过（更快） |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | ON | 生成 `compile_commands.json` 供 clangd/IDE 使用 |

### 4. 编译

```bash
ninja -j$(sysctl -n hw.ncpu)
```

macOS 使用 `sysctl -n hw.ncpu` 获取 CPU 核数（Linux 用 `nproc`）。

首次全量编译约需 30-60 分钟（取决于机器配置和 `WITH_TESTING` 开关）。后续增量编译通常几秒到几分钟。

### 5. 安装

```bash
cd /path/to/Paddle
uv pip install build/python/dist/*.whl --no-deps --force-reinstall
```

### 6. 验证

```bash
python -c "import paddle; print(paddle.__version__); paddle.utils.run_check()"
```

## 一键脚本

```bash
#!/bin/bash
set -e

PY_VERSION=${1:-3.10}
PADDLE_DIR=$(pwd)
VENV_DIR=${PADDLE_DIR}/venvs/paddle-py${PY_VERSION//./}

# 环境准备
if [ ! -d ${VENV_DIR} ]; then
    uv venv --seed ${VENV_DIR} --python=${PY_VERSION}
fi
source ${VENV_DIR}/bin/activate
uv pip install -r python/requirements.txt

# Python 路径
PY_LIB_DIR=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PY_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")

# CMake 配置 + 编译
mkdir -p build && cd build
rm -rf CMakeCache.txt CMakeFiles/
cmake .. \
    -GNinja \
    -DPY_VERSION=${PY_VERSION} \
    -DPYTHON_INCLUDE_DIR=${PY_INCLUDE_DIR} \
    -DPYTHON_LIBRARY=${PY_LIB_DIR}/libpython${PY_VERSION}.dylib \
    -DWITH_GPU=OFF \
    -DWITH_ARM=ON \
    -DWITH_AVX=OFF \
    -DWITH_TESTING=OFF \
    -DWITH_DISTRIBUTE=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja -j$(sysctl -n hw.ncpu)

# 安装
cd ${PADDLE_DIR}
uv pip install build/python/dist/*.whl --no-deps --force-reinstall
python -c "import paddle; print(paddle.__version__)"
```

## macOS 特有的常见问题

| 症状 | 原因 | 解决 |
|------|------|------|
| `Ignoring CMAKE_OSX_SYSROOT` + 配置失败 | pip 安装的 cmake 无法找到系统 SDK | 用 `brew install cmake` 的系统 cmake，不要用 venv 中的 |
| `failed to recurse into submodule` | git worktree 残留导致 submodule config 中路径失效 | 修复 `.git/modules/*/config` 中的 `worktree` 路径，或 `git submodule deinit -f . && git submodule update --init --recursive` |
| `CMakeCache.txt directory is different` | build 目录从别的路径（如 worktree）复制过来 | `rm -rf CMakeCache.txt CMakeFiles/` 清理缓存后重新 cmake |
| `WITH_AVX` 相关编译错误 | 未设置 `WITH_ARM=ON` 导致尝试使用 x86 指令集 | 添加 `-DWITH_ARM=ON -DWITH_AVX=OFF` |
| 链接时 `symbol not found` | SDK 或系统库版本不匹配 | 确保 Xcode CLT 是最新版本：`xcode-select --install`，并检查 `SDKROOT` 设置（如 `export SDKROOT=$(xcrun --show-sdk-path)`） |
| `libpython*.dylib not found` | Python 路径不正确 | 用 `sysconfig` 动态获取路径，不要硬编码 |

## 与 Linux 编译的主要差异

| 方面 | Linux (x86_64 + CUDA) | macOS (Apple Silicon) |
|------|----------------------|----------------------|
| 架构标志 | 默认 x86_64 | `-DWITH_ARM=ON -DWITH_AVX=OFF` |
| GPU | `-DWITH_GPU=ON` | `-DWITH_GPU=OFF` |
| BLAS | MKL | Apple Accelerate（自动检测） |
| CPU 核数 | `nproc` | `sysctl -n hw.ncpu` |
| Python lib | `.so` | `.dylib` |
| 分布式 | NCCL 支持 | 不支持 |
