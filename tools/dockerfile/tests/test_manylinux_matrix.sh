#!/usr/bin/env bash

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
