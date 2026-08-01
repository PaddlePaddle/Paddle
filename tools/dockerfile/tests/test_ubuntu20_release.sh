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
