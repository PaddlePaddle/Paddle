# C++20 Release 镜像生成方案

## 目标

在一个改动中解决 PaddlePaddle/Paddle#79384 的第 5、6 项，移除仍会选择 GCC 8.2 的 Release 镜像生成路径。

## 方案对比

1. 退役旧 CUDA 11 和 CentOS 生成器。这是改动最小的方案，并直接复用现有的新版 manylinux 定义。
2. 保留 CUDA 11 变体并强制使用 GCC 11/12。这样可以保留旧产物，但会引入不受支持的宿主编译器组合，尤其是 CUDA 11.2。
3. 保留旧生成器，并注明它们不属于 C++20 检查范围。这样可以保留兼容性，但无法完成 issue 中跟踪的升级目标。

采用方案 1。

## 改动范围

- 在 `tools/dockerfile/ubuntu20_release.sh` 中停止生成 CUDA 11.2 和 CUDA 11.8 Dockerfile。
- 保留 CPU Release 镜像，但不再把 `Dockerfile.release.ubuntu20` 中的 GCC 12.1 改回 GCC 8.2。
- 保持现有 CUDA 12.0、12.3 和 12.6 Release 变体不变。
- 删除 `tools/dockerfile/centos7_manylinux.sh` 及其没有其他引用的 `tools/dockerfile/Dockerfile.centos` 模板。
- 使用已经采用 AlmaLinux 8 和 `gcc-toolset-11` 的 `tools/dockerfile/manylinux/Dockerfile` 作为继续维护的 manylinux 路径。

不修改 CMake 标准开关、新版 manylinux 文件、CUDA 12 镜像版本或 Release 发布工作流。

## 验证

- 修改前运行 shell 断言，确认旧文件和 GCC 8.2 产物仍存在时检查会失败。
- 运行 `bash -n tools/dockerfile/ubuntu20_release.sh`。
- 在临时目录生成 Ubuntu Release Dockerfile，并确认：
  - 不再生成 CUDA 11.2 和 CUDA 11.8 产物；
  - 仍生成 CPU、CUDA 12.0、12.3 和 CUDA 12.6 产物；
  - 所有生成产物均使用 GCC 12.1，且不包含 GCC 8.2 引用。
- 确认仓库中不再引用已删除的 CentOS 文件。
- 运行 `git diff --check` 和文件范围的仓库检查。

Docker 镜像的实际构建和发布仍交由 CI 验证，因为它依赖 Release 镜像基础设施和外部构建产物。
