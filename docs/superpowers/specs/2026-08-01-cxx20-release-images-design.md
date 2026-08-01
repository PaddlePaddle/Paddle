# C++20 Release 镜像清理方案

## 目标

在一个改动中解决 PaddlePaddle/Paddle#79384 的第 5、6 项：移除仍会选择 GCC 8.2 或 CUDA 12.8 以下版本的 Release/manylinux 生成路径，并补齐仍需维护的 CUDA 12.8 镜像。

## 最终支持矩阵

| 路径 | 保留的变体 | 不支持的变体 |
| --- | --- | --- |
| `tools/dockerfile/ubuntu20_release.sh` | CPU、CUDA 12.8、CUDA 12.9 | CUDA 13.x 及其他版本继续进入现有错误分支 |
| `tools/dockerfile/manylinux/Dockerfile` | CUDA 12.8、12.9、13.0、13.2 | CUDA 12.8 以下版本，以及未纳入 Paddle 当前矩阵的 13.1/13.3 |
| `tools/dockerfile/manylinux/Dockerfile-*` | `Dockerfile-128/-129/-130/-132` | 删除 `Dockerfile-118/-126`，不新增 `Dockerfile-133` |

CUDA 13.0 及以上没有 Ubuntu 20.04 官方基础镜像，因此 Ubuntu 20 Release 生成器只支持 CUDA 12.8 和 12.9。CUDA 13.2 继续由 manylinux 路径提供。

## 方案选择

1. 完整清理相关生成路径：同步更新 Ubuntu 20、通用 manylinux、独立 manylinux Dockerfile 和共用安装脚本，并删除废弃 CentOS 路径。
2. 只修改两个主要入口：改动更少，但会留下 CUDA 11/12.6 的独立 Dockerfile 和已失效的 CentOS 生成器，与版本政策矛盾。
3. 同时扩展 Paddle CUDA 依赖元数据和源码适配：会把镜像清理扩大为 CUDA 平台适配，不属于 issue #79384 这两个条目的范围。

采用方案 1。

## Ubuntu 20 Release 生成器

- 保留 `tools/dockerfile/ubuntu20_release.sh` 和 `Dockerfile.release.ubuntu20` 的名称，不把 CUDA 13.x 混入 Ubuntu 20 路径。
- GPU 只生成：
  - `Dockerfile-128`：`nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04`；
  - `Dockerfile-129`：`nvidia/cuda:12.9.0-cudnn-devel-ubuntu20.04`。
- GPU 产物直接使用基础镜像自带的 cuDNN，不再运行旧的手工 cuDNN/TensorRT 安装步骤。
- CPU 产物继续使用 `ubuntu:20.04`，但不再把 GCC 12.1 降级为 GCC 8.2。
- 为 GCC 12.1 显式设置编译器和运行库搜索路径，兼容 PR #79600 的 side-by-side GCC 安装契约。
- 保留未知 `ref_CUDA_MAJOR` 打印 `Dockerfile ERROR!!!` 并返回非零状态的现有错误分支；不新增 CUDA 13.x 分支。

## manylinux 路径

- 删除 `tools/dockerfile/centos7_manylinux.sh` 及其唯一模板 `tools/dockerfile/Dockerfile.centos`。该生成器没有仓库内调用者，且仍包含 GCC 8.2 和已不存在的 RockyLinux 模板引用。
- `tools/dockerfile/manylinux/Dockerfile` 的默认 CUDA 从 11.8 改为 12.8。
- 删除 CUDA 11.8、12.3、12.4、12.6 build stage；保留 12.9、13.0、13.2，并新增 12.8 stage。
- CUDA 12.8 stage 使用现有安装模式：
  - CUDA 12.8.1；
  - cuDNN 9.7.1.26；
  - NCCL 2.25.1 的 CUDA 12.8 包；
  - cuSparseLt 0.6.3；
  - 复用现有 GDRCopy 安装脚本。
- `tools/dockerfile/manylinux/common/install_cuda.sh` 删除仅供 CUDA 12.8 以下版本使用的 installer、prune 函数和 dispatcher 分支。
- `install_nccl_2234`、`install_cusparselt_063` 等仍被 CUDA 12.9 使用的 helper 必须保留，不能按名称中的旧版本误删。
- 删除已经没有调用者的 TensorRT helper，不为 CUDA 12.8 或 CUDA 13.x 恢复 TensorRT 安装。
- 删除独立的 `Dockerfile-118/-126`，新增以 `Dockerfile-129` 结构为基线的 `Dockerfile-128`；保留 `Dockerfile-129/-130/-132`。

## 并行改动处理

开放中的 PR #79600 同时修改 `ubuntu20_release.sh`、`centos7_manylinux.sh` 和 GCC 运行库选择。实现时以本方案的新版本矩阵为准：

- 被删除的 CentOS 和旧 CUDA 分支不保留 #79600 针对它们的补丁；
- Ubuntu CPU/GCC 12.1 路径保留显式 `LD_LIBRARY_PATH`，避免 side-by-side 安装后回落到系统旧运行库；
- 创建 PR 前重新基于最新 `develop` 检查 #79600 的合入状态并解决文本冲突。

## 非目标

- 不支持 CUDA 13.3，也不新增 CUDA 13.1。
- 不修改 `setup.py`、`python/setup.py.in`、CUDA CMake 标准或 Paddle 源码兼容逻辑。
- 不修改发布 workflow、镜像仓库标签或下游项目配置。
- 不在本地发布镜像。

## 验证

- 运行 `bash -n` 检查 `ubuntu20_release.sh` 和 `manylinux/common/install_cuda.sh`。
- 在临时目录运行 Ubuntu 20 生成器，确认只生成 CPU、CUDA 12.8 和 CUDA 12.9 产物。
- 检查生成产物：
  - GPU 基础镜像标签与矩阵一致；
  - 不包含 CUDA 12.8 以下版本、GCC 8.2、旧 TensorRT 或手工 cuDNN 安装；
  - CPU 产物使用 GCC 12.1，并包含对应运行库路径。
- 检查 manylinux 通用 Dockerfile 的默认目标和 stage 恰好为 12.8、12.9、13.0、13.2。
- 检查 `install_cuda.sh` dispatcher 恰好接受上述版本，未知版本返回非零状态。
- 验证 CUDA 12.8 基础镜像和安装包 URL 可访问。
- 确认仓库中不再引用已删除的 CentOS 文件和 CUDA 11/12.6 独立 Dockerfile。
- 运行 `git diff --check` 和目标文件范围的 `prek` 检查。

完整 Docker 镜像构建和 Paddle 编译依赖 Linux x86_64 构建环境与外部镜像基础设施，最终由镜像 CI 验证。
