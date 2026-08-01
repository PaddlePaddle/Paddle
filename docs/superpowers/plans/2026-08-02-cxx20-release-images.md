# C++20 Release 镜像清理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清理 CUDA 12.8 以下的 Ubuntu Release/CentOS/manylinux 生成路径，让 Ubuntu 20 只生成 CPU、CUDA 12.8、12.9，让 manylinux 只支持 CUDA 12.8、12.9、13.0、13.2。

**Architecture:** Ubuntu 20 继续使用现有模板生成三个 Dockerfile，但删除旧 cuDNN/TensorRT 覆盖与 GCC 8.2 降级。manylinux 继续使用 AlmaLinux 多阶段 Dockerfile 和共用安装脚本，新增 CUDA 12.8 stage，删除旧 stage、installer、prune 逻辑和无调用者的 TensorRT helper。两个小型 Shell 回归检查分别锁定生成结果和 manylinux 版本矩阵。

**Tech Stack:** Bash、GNU sed、Dockerfile/BuildKit、NVIDIA CUDA 镜像与 RPM/tar 包、prek。

## Global Constraints

- Ubuntu 20 Release 只保留 CPU、CUDA 12.8、CUDA 12.9；CUDA 13.x 继续进入现有错误分支。
- manylinux 只保留 CUDA 12.8、12.9、13.0、13.2；不新增 CUDA 13.1 或 13.3。
- CPU Release 使用 GCC 12.1，不再降级到 GCC 8.2，并显式选择 GCC 12.1 运行库。
- 删除 `centos7_manylinux.sh`、`Dockerfile.centos`、`manylinux/Dockerfile-118`、`manylinux/Dockerfile-126`。
- 不修改 `setup.py`、`python/setup.py.in`、CUDA CMake 标准、workflow 或下游镜像标签。
- 不恢复 TensorRT 安装；CUDA 12.8/12.9 Ubuntu 镜像直接使用基础镜像自带 cuDNN。
- 保留所有与本任务无关的未跟踪文件；每次提交只暂存任务明确列出的路径。

## 文件结构

- Create: `tools/dockerfile/tests/test_ubuntu20_release.sh` — 验证 Ubuntu 20 生成结果和旧 CentOS 路径退役。
- Create: `tools/dockerfile/tests/test_manylinux_matrix.sh` — 验证 manylinux 默认版本、stage、dispatcher 和独立 Dockerfile 矩阵。
- Create: `tools/dockerfile/manylinux/Dockerfile-128` — CUDA 12.8 独立 Ubuntu 22.04 构建镜像。
- Modify: `tools/dockerfile/ubuntu20_release.sh:17-101` — 收敛为 CPU/12.8/12.9。
- Modify: `tools/dockerfile/Dockerfile.release.ubuntu20:20-55` — 删除旧 NVIDIA 源、TensorRT/手工 cuDNN，固定 GCC 12.1 runtime。
- Modify: `tools/dockerfile/manylinux/Dockerfile:1-107` — 默认改为 12.8，收敛 build stage。
- Modify: `tools/dockerfile/manylinux/common/install_cuda.sh:22-512` — 新增 12.8 installer，删除旧版本和死代码。
- Delete: `tools/dockerfile/centos7_manylinux.sh`。
- Delete: `tools/dockerfile/Dockerfile.centos`。
- Delete: `tools/dockerfile/manylinux/Dockerfile-118`。
- Delete: `tools/dockerfile/manylinux/Dockerfile-126`。

---

### Task 1: 收敛 Ubuntu 20 Release 生成器

**Files:**
- Create: `tools/dockerfile/tests/test_ubuntu20_release.sh`
- Modify: `tools/dockerfile/ubuntu20_release.sh:17-101`
- Modify: `tools/dockerfile/Dockerfile.release.ubuntu20:20-55`
- Delete: `tools/dockerfile/centos7_manylinux.sh`
- Delete: `tools/dockerfile/Dockerfile.centos`

**Interfaces:**
- Consumes: `ubuntu20_release.sh` 从当前工作目录读取 `Dockerfile.release.ubuntu20`。
- Produces: `Dockerfile-cpu`、`Dockerfile-128`、`Dockerfile-129`；其他 CUDA 版本由 `base_image` 返回非零状态。

- [ ] **Step 1: 写 Ubuntu Release 失败检查**

使用 `apply_patch` 创建可执行文件 `tools/dockerfile/tests/test_ubuntu20_release.sh`：

```bash
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

cp "${DOCKERFILE_DIR}/ubuntu20_release.sh" "${TMP_DIR}/"
cp "${DOCKERFILE_DIR}/Dockerfile.release.ubuntu20" "${TMP_DIR}/"
(
  cd "${TMP_DIR}"
  bash ubuntu20_release.sh
)

expected_files="Dockerfile-128 Dockerfile-129 Dockerfile-cpu"
actual_files="$(find "${TMP_DIR}" -maxdepth 1 -type f -name 'Dockerfile-*' -exec basename {} \; | sort | paste -sd ' ' -)"
[[ "${actual_files}" == "${expected_files}" ]] || {
  echo "unexpected Ubuntu Release files: ${actual_files}"
  exit 1
}

grep -Fq 'FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04' "${TMP_DIR}/Dockerfile-128"
grep -Fq 'FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu20.04' "${TMP_DIR}/Dockerfile-129"
grep -Fq 'FROM ubuntu:20.04' "${TMP_DIR}/Dockerfile-cpu"
grep -Fq 'ENV PATH=/usr/local/gcc-12.1/bin:$PATH' "${TMP_DIR}/Dockerfile-cpu"
grep -Fq 'ENV LD_LIBRARY_PATH=/usr/local/gcc-12.1/lib64:/usr/local/gcc-12.1/lib:${LD_LIBRARY_PATH}' "${TMP_DIR}/Dockerfile-cpu"

if grep -ERq 'gcc-8\.2|gcc82|cuda-(11|12\.[0-7])|install_trt\.sh|install_cudnn\.sh' \
  "${TMP_DIR}/Dockerfile-128" "${TMP_DIR}/Dockerfile-129" "${TMP_DIR}/Dockerfile-cpu"; then
  echo 'legacy Ubuntu Release dependency found'
  exit 1
fi

grep -Fq 'echo "Dockerfile ERROR!!!"' "${DOCKERFILE_DIR}/ubuntu20_release.sh"
[[ ! -e "${DOCKERFILE_DIR}/centos7_manylinux.sh" ]]
[[ ! -e "${DOCKERFILE_DIR}/Dockerfile.centos" ]]
```

然后设置执行位：

```bash
chmod +x tools/dockerfile/tests/test_ubuntu20_release.sh
```

- [ ] **Step 2: 运行检查并确认失败原因**

Run:

```bash
bash tools/dockerfile/tests/test_ubuntu20_release.sh
```

Expected: FAIL；生成文件仍是 `Dockerfile-112/-118/-120/-123/-126/-cpu`，且旧 CentOS 文件仍存在。

- [ ] **Step 3: 最小化 Ubuntu 20 模板**

使用 `apply_patch` 修改 `Dockerfile.release.ubuntu20`：

- 删除三条硬编码 Ubuntu 20 NVIDIA apt 源的命令；官方 CUDA 基础镜像已经配置匹配源，CPU 基础镜像不需要 NVIDIA 源。
- 删除 `RUN bash /build_scripts/install_trt.sh`。
- 删除 `RUN bash /build_scripts/install_cudnn.sh cudnn841` 和 `ENV CUDNN_VERSION=8.4.1`。
- 将 `# Downgrade gcc&&g++` 改为 `# Install GCC 12.1`。
- 在 GCC `PATH` 后加入：

```dockerfile
ENV LD_LIBRARY_PATH=/usr/local/gcc-12.1/lib64:/usr/local/gcc-12.1/lib:${LD_LIBRARY_PATH}
```

- 将错误的 `RUN rm -rf /build_script` 改为：

```dockerfile
RUN rm -rf /build_scripts
```

- [ ] **Step 4: 把生成分支收敛到 CPU/12.8/12.9**

使用 `apply_patch` 删除 `ubuntu20_release.sh` 的 11.2、11.8、12.0、12.3、12.6 分支，保留现有 `else` 错误处理，并加入以下两个 GPU 分支：

```bash
if [[ ${ref_CUDA_MAJOR} == "12.8" ]]; then
  dockerfile_name="Dockerfile-128"
  sed "s#<baseimg>#nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04#g" ./Dockerfile.release.ubuntu20 >${dockerfile_name}
  sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
  sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
elif [[ ${ref_CUDA_MAJOR} == "12.9" ]]; then
  dockerfile_name="Dockerfile-129"
  sed "s#<baseimg>#nvidia/cuda:12.9.0-cudnn-devel-ubuntu20.04#g" ./Dockerfile.release.ubuntu20 >${dockerfile_name}
  sed -i "s#<setcuda>#ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH #g" ${dockerfile_name}
  sed -i 's#<install_cpu_package>##g' ${dockerfile_name}
```

CPU 分支只替换 base image、CUDA 占位符、CPU 包和 `WITH_GPU` 默认值；删除所有 GCC 8.2、TensorRT、cuDNN、NVIDIA apt 源相关 `sed`。文件末尾调用顺序固定为：

```bash
export ref_CUDA_MAJOR=0
base_image
export ref_CUDA_MAJOR=12.8
base_image
export ref_CUDA_MAJOR=12.9
base_image
```

- [ ] **Step 5: 删除废弃 CentOS 生成路径**

使用 `apply_patch` 删除：

```text
tools/dockerfile/centos7_manylinux.sh
tools/dockerfile/Dockerfile.centos
```

- [ ] **Step 6: 运行 Ubuntu 检查并确认通过**

Run:

```bash
bash -n tools/dockerfile/ubuntu20_release.sh
bash tools/dockerfile/tests/test_ubuntu20_release.sh
git grep -nE 'centos7_manylinux\.sh|Dockerfile\.centos' -- ':!docs/superpowers/**'
```

Expected: 前两条命令 PASS；最后一条无输出并以 1 退出，证明没有剩余 tracked 引用。

- [ ] **Step 7: 提交 Ubuntu/CentOS 清理**

```bash
git add -- \
  tools/dockerfile/tests/test_ubuntu20_release.sh \
  tools/dockerfile/ubuntu20_release.sh \
  tools/dockerfile/Dockerfile.release.ubuntu20 \
  tools/dockerfile/centos7_manylinux.sh \
  tools/dockerfile/Dockerfile.centos
git commit -m "docker: update Ubuntu release image matrix"
```

### Task 2: 收敛 manylinux CUDA 矩阵

**Files:**
- Create: `tools/dockerfile/tests/test_manylinux_matrix.sh`
- Create: `tools/dockerfile/manylinux/Dockerfile-128`
- Modify: `tools/dockerfile/manylinux/Dockerfile:1-107`
- Modify: `tools/dockerfile/manylinux/common/install_cuda.sh:22-512`
- Delete: `tools/dockerfile/manylinux/Dockerfile-118`
- Delete: `tools/dockerfile/manylinux/Dockerfile-126`

**Interfaces:**
- Consumes: Docker build argument `CUDA_VERSION` and target name `cuda${CUDA_VERSION}`。
- Produces: `cuda12.8`、`cuda12.9`、`cuda13.0`、`cuda13.2` stage；`install_cuda.sh` 接受同一组版本字符串。

- [ ] **Step 1: 写 manylinux 失败检查**

使用 `apply_patch` 创建可执行文件 `tools/dockerfile/tests/test_manylinux_matrix.sh`：

```bash
#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANYLINUX_DIR="${DOCKERFILE_DIR}/manylinux"
DOCKERFILE="${MANYLINUX_DIR}/Dockerfile"
INSTALL_CUDA="${MANYLINUX_DIR}/common/install_cuda.sh"

grep -Fxq 'ARG CUDA_VERSION=12.8' "${DOCKERFILE}"

expected_stages='12.8 12.9 13.0 13.2'
actual_stages="$(sed -n 's/^FROM cuda as cuda//p' "${DOCKERFILE}" | paste -sd ' ' -)"
[[ "${actual_stages}" == "${expected_stages}" ]] || {
  echo "unexpected manylinux stages: ${actual_stages}"
  exit 1
}

expected_files='Dockerfile-128 Dockerfile-129 Dockerfile-130 Dockerfile-132'
actual_files="$(find "${MANYLINUX_DIR}" -maxdepth 1 -type f -name 'Dockerfile-[0-9]*' -exec basename {} \; | sort | paste -sd ' ' -)"
[[ "${actual_files}" == "${expected_files}" ]] || {
  echo "unexpected standalone manylinux Dockerfiles: ${actual_files}"
  exit 1
}

grep -Fq 'FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 as base' "${MANYLINUX_DIR}/Dockerfile-128"
grep -Eq '^function install_128([[:space:]]|\{)' "${INSTALL_CUDA}"
grep -Eq '^[[:space:]]*12\.8\)[[:space:]]+install_128' "${INSTALL_CUDA}"
grep -Eq '^[[:space:]]*12\.9\)[[:space:]]+install_129' "${INSTALL_CUDA}"
grep -Eq '^[[:space:]]*13\.0\)[[:space:]]+install_130' "${INSTALL_CUDA}"
grep -Eq '^[[:space:]]*13\.2\)[[:space:]]+install_132' "${INSTALL_CUDA}"
grep -Eq '^function install_nccl_2234([[:space:]]|\{)' "${INSTALL_CUDA}"
grep -Eq '^function install_cusparselt_063([[:space:]]|\{)' "${INSTALL_CUDA}"

expected_cases='12.8 12.9 13.0 13.2'
actual_cases="$(sed -n 's/^[[:space:]]*\([0-9][0-9.]*\))[[:space:]].*/\1/p' "${INSTALL_CUDA}" | paste -sd ' ' -)"
[[ "${actual_cases}" == "${expected_cases}" ]] || {
  echo "unexpected install_cuda cases: ${actual_cases}"
  exit 1
}

if grep -Eq '^function (install|prune)_(118|123|124|126)([[:space:]]|\{)' "${INSTALL_CUDA}"; then
  echo 'legacy CUDA installer or pruner found'
  exit 1
fi

if grep -Eq '^function install_trt_' "${INSTALL_CUDA}"; then
  echo 'unused TensorRT helper found'
  exit 1
fi

if bash "${INSTALL_CUDA}" 13.3; then
  echo 'unsupported CUDA 13.3 unexpectedly succeeded'
  exit 1
fi
```

然后设置执行位：

```bash
chmod +x tools/dockerfile/tests/test_manylinux_matrix.sh
```

- [ ] **Step 2: 运行检查并确认失败原因**

Run:

```bash
bash tools/dockerfile/tests/test_manylinux_matrix.sh
```

Expected: FAIL；默认仍为 11.8，stage 和独立 Dockerfile 仍包含 11.8/12.6，且缺少 12.8。

- [ ] **Step 3: 新增 CUDA 12.8 安装函数**

在 `manylinux/common/install_cuda.sh` 保留 `install_cusparselt_063`，新增 NCCL helper：

```bash
function install_nccl_2251_cuda128 {
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    yum install -y \
        cuda-compat-12-8 \
        libnccl-2.25.1-1+cuda12.8 \
        libnccl-devel-2.25.1-1+cuda12.8 \
        libnccl-static-2.25.1-1+cuda12.8
}
```

新增 installer：

```bash
function install_128 {
    CUDNN_VERSION=9.7.1.26
    NCCL_VERSION=2.25.1
    echo "Installing CUDA 12.8.1 and cuDNN ${CUDNN_VERSION} and NCCL ${NCCL_VERSION} and cuSparseLt-0.6.3"
    rm -rf /usr/local/cuda-12.8 /usr/local/cuda
    wget -q https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
    chmod +x cuda_12.8.1_570.124.06_linux.run
    ./cuda_12.8.1_570.124.06_linux.run --toolkit --silent
    rm -f cuda_12.8.1_570.124.06_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-12.8 /usr/local/cuda

    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    tar xf cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/include/* /usr/local/cuda/include/
    cp -a cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive/lib/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn

    install_nccl_2251_cuda128
    install_cusparselt_063
    ldconfig
}
```

- [ ] **Step 4: 删除旧 installer、prune 和 TensorRT 死代码**

使用 `apply_patch` 删除以下仅供被退役版本使用的函数：

```text
install_cusparselt_040
install_cusparselt_052
install_cusparselt_062
install_nccl_2162
install_nccl_2203
install_nccl_2215
install_118
install_123
install_124
install_126
prune_118
prune_123
prune_124
prune_126
install_trt_8616
install_trt_105018
install_trt_1016111
install_trt_101339
```

必须保留 `install_nccl_2234` 和 `install_cusparselt_063`，因为 `install_129` 仍调用它们。dispatcher 最终只包含：

```bash
12.8) install_128
    ;;
12.9) install_129
    ;;
13.0) install_130
    ;;
13.2) install_132
    ;;
*) echo "bad argument $1"; exit 1
    ;;
```

- [ ] **Step 5: 收敛通用 manylinux Dockerfile stage**

将默认版本改为：

```dockerfile
ARG CUDA_VERSION=12.8
```

删除 `cuda11.8`、`cuda12.3`、`cuda12.4`、`cuda12.6` stage，并在 `cuda12.9` 前加入：

```dockerfile
FROM cuda as cuda12.8
RUN bash ./install_cuda.sh 12.8 && bash ./install_gdrcopy.sh
ENV DESIRED_CUDA=12.8
ENV GDRCOPY_HOME=/usr/local/gdrcopy
```

最终 stage 顺序必须为 12.8、12.9、13.0、13.2，`FROM ${BASE_TARGET} as final` 保持不变。

- [ ] **Step 6: 调整独立 manylinux Dockerfile**

使用 `apply_patch` 删除：

```text
tools/dockerfile/manylinux/Dockerfile-118
tools/dockerfile/manylinux/Dockerfile-126
```

创建 `Dockerfile-128`，正文沿用当前 `Dockerfile-129` 的单层构建结构，只使用以下 CUDA 12.8 定义：

```dockerfile
ARG CUDA_VERSION=12.8
ARG BASE_TARGET=cuda${CUDA_VERSION}

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 as base
```

其余系统包、CMake、patchelf、ccache 和 Python requirements 安装步骤与当前 `Dockerfile-129` 保持一致；不新增 NCCL/TensorRT 手工安装。

- [ ] **Step 7: 运行 manylinux 检查并确认通过**

Run:

```bash
bash -n tools/dockerfile/manylinux/common/install_cuda.sh
bash tools/dockerfile/tests/test_manylinux_matrix.sh
```

Expected: PASS；对 CUDA 13.3 的子检查输出 `bad argument 13.3` 后仍整体 PASS。

- [ ] **Step 8: 提交 manylinux 清理**

```bash
git add -- \
  tools/dockerfile/tests/test_manylinux_matrix.sh \
  tools/dockerfile/manylinux/Dockerfile \
  tools/dockerfile/manylinux/Dockerfile-118 \
  tools/dockerfile/manylinux/Dockerfile-126 \
  tools/dockerfile/manylinux/Dockerfile-128 \
  tools/dockerfile/manylinux/common/install_cuda.sh
git commit -m "docker: clean up manylinux CUDA matrix"
```

### Task 3: 集成验证和冲突复核

**Files:**
- Verify only: all files changed by Task 1 and Task 2

**Interfaces:**
- Consumes: Task 1 的三个 Ubuntu 生成产物契约，以及 Task 2 的四版本 manylinux 契约。
- Produces: 可提交审查的静态验证结果；完整镜像构建明确留给 Linux 镜像 CI。

- [ ] **Step 1: 运行两个回归检查**

```bash
bash tools/dockerfile/tests/test_ubuntu20_release.sh
bash tools/dockerfile/tests/test_manylinux_matrix.sh
```

Expected: 两者 PASS。

- [ ] **Step 2: 验证官方镜像和 CUDA 12.8 安装包可访问**

```bash
docker manifest inspect nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04 >/dev/null
docker manifest inspect nvidia/cuda:12.9.0-cudnn-devel-ubuntu20.04 >/dev/null
docker manifest inspect nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 >/dev/null
curl -fsSIL https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run >/dev/null
curl -fsSIL https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.7.1.26_cuda12-archive.tar.xz >/dev/null
curl -fsSIL https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.6.3.2-archive.tar.xz >/dev/null
```

Expected: 全部返回 0。

- [ ] **Step 3: 运行 Dockerfile 静态检查**

```bash
docker buildx build --check --target cuda12.8 \
  -f tools/dockerfile/manylinux/Dockerfile \
  tools/dockerfile/manylinux
docker buildx build --check \
  -f tools/dockerfile/manylinux/Dockerfile-128 \
  .
```

Expected: 两个 Dockerfile 均通过 BuildKit 静态检查；不执行完整镜像构建。

- [ ] **Step 4: 运行仓库检查**

```bash
prek run --files \
  tools/dockerfile/tests/test_ubuntu20_release.sh \
  tools/dockerfile/tests/test_manylinux_matrix.sh \
  tools/dockerfile/ubuntu20_release.sh \
  tools/dockerfile/Dockerfile.release.ubuntu20 \
  tools/dockerfile/manylinux/Dockerfile \
  tools/dockerfile/manylinux/Dockerfile-128 \
  tools/dockerfile/manylinux/common/install_cuda.sh
git diff --check upstream/develop...HEAD
```

Expected: PASS。prek 的本地无效缓存 warning 可以记录，但不能掩盖 hook failure。

- [ ] **Step 5: 复核 PR #79600 的并行状态和最终范围**

```bash
gh pr view 79600 --repo PaddlePaddle/Paddle \
  --json state,headRefOid,mergeStateStatus,reviewDecision,url
git diff --name-status upstream/develop...HEAD
git status --short
```

Expected: 记录 #79600 是否已合入；最终 diff 只包含设计/计划文档及本计划列出的 Docker/test 文件，用户原有未跟踪文件保持未暂存。

- [ ] **Step 6: 如验证修复产生新改动，单独提交**

仅当 Step 1-5 暴露问题并实际修改了目标文件时执行：

```bash
git add -- \
  tools/dockerfile/tests/test_ubuntu20_release.sh \
  tools/dockerfile/tests/test_manylinux_matrix.sh \
  tools/dockerfile/ubuntu20_release.sh \
  tools/dockerfile/Dockerfile.release.ubuntu20 \
  tools/dockerfile/manylinux/Dockerfile \
  tools/dockerfile/manylinux/Dockerfile-128 \
  tools/dockerfile/manylinux/common/install_cuda.sh
git commit -m "test: verify release image matrix"
```

若验证未产生改动，不创建空提交。
