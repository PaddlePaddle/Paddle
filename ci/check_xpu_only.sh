#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#
# check_xpu_only.sh - 判断 PR 是否为纯 XPU 改动
#
# 当 PR 变更文件全部属于 XPU 专属路径时，其他硬件 CI 可以跳过。
#
# Exit codes:
#   0 = 纯 XPU 改动（其他硬件 CI 可跳过）
#   1 = 非纯 XPU 改动（所有 CI 照常运行）
#
# Safety: 任何异常默认 exit 1（全量运行），绝不错跳。
#

set -o pipefail

LOG_PREFIX="[xpu-only]"

# =============================================================================
# 紧急开关：全局禁用跳过
# =============================================================================
if [[ "${DISABLE_HW_CI_SKIP:-false}" == "true" ]]; then
    echo "$LOG_PREFIX Emergency override: DISABLE_HW_CI_SKIP=true, running all CI"
    exit 1
fi

# =============================================================================
# 强制全量运行：commit message 含 test=all
# =============================================================================
COMMIT_MSG=$(git log -1 --pretty=%B 2>/dev/null || echo "")
if echo "$COMMIT_MSG" | grep -qiE 'test[=:] *all'; then
    echo "$LOG_PREFIX Force full CI: commit message contains test=all"
    exit 1
fi

# =============================================================================
# 获取变更文件
# =============================================================================

CHANGED_FILES=""

# Method 1: GitHub PR 上下文（对比 base branch）
if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    CHANGED_FILES=$(git diff --name-only "origin/${GITHUB_BASE_REF}...HEAD" 2>/dev/null || echo "")
fi

# Method 2: 对比 develop 分支
if [[ -z "$CHANGED_FILES" ]]; then
    CHANGED_FILES=$(git diff --name-only origin/develop...HEAD 2>/dev/null || echo "")
fi

# Method 3: 对比上一个 commit（兜底）
if [[ -z "$CHANGED_FILES" ]]; then
    CHANGED_FILES=$(git diff --name-only HEAD~1 2>/dev/null || echo "")
fi

# 无法获取变更文件 → 全量运行
if [[ -z "$CHANGED_FILES" ]]; then
    echo "$LOG_PREFIX WARN: No changed files detected, defaulting to full CI"
    exit 1
fi

TOTAL_FILES=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
echo "$LOG_PREFIX Analyzing $TOTAL_FILES changed files"

# =============================================================================
# XPU 专属路径定义
# =============================================================================
XPU_PATHS=(
    # phi kernels
    "paddle/phi/kernels/xpu/"
    "paddle/phi/kernels/fusion/xpu/"
    "paddle/phi/kernels/selected_rows/xpu/"
    "paddle/phi/kernels/sparse/xpu/"
    "paddle/phi/kernels/legacy/xpu/"
    # phi backends & device
    "paddle/phi/backends/xpu/"
    "paddle/phi/core/platform/device/xpu/"
    # fluid (旧框架遗留)
    "paddle/fluid/pir/transforms/xpu/"
    "paddle/fluid/framework/ir/xpu/"
    "paddle/fluid/platform/device/xpu/"
    # python API
    "python/paddle/device/xpu/"
    "python/paddle/incubate/xpu/"
    # tests
    "test/xpu/"
    "test/cpp/fluid/framework/ir/xpu/"
    "test/ir/pir/fused_pass/xpu/"
    # build & tools
    "cmake/external/xpu.cmake"
    "cmake/xpu_kp.cmake"
    "tools/xpu/"
)

# =============================================================================
# 逐文件判断：是否全部属于 XPU 专属路径
# =============================================================================
while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    is_xpu=false
    for path in "${XPU_PATHS[@]}"; do
        if [[ "$file" == "$path"* ]]; then
            is_xpu=true
            break
        fi
    done

    if [[ "$is_xpu" == "false" ]]; then
        echo "$LOG_PREFIX Non-XPU file found: $file"
        echo "$LOG_PREFIX Result: NOT xpu-only PR, running all CI"
        exit 1
    fi
done <<< "$CHANGED_FILES"

echo "$LOG_PREFIX All $TOTAL_FILES changed files are XPU-specific"
echo "$LOG_PREFIX Result: XPU-only PR, other hardware CI can be skipped"
exit 0
