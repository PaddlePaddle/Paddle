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
# check_xpu_only.sh - Determine whether a PR only modifies XPU-specific files
#
# When all changed files in a PR belong to XPU-specific paths,
# other hardware CI jobs (GPU/DCU/NPU) can be safely skipped.
#
# Exit codes:
#   0 = XPU-only changes (other hardware CI can be skipped)
#   1 = Non-XPU-only changes (all CI should run as usual)
#
# Safety: Any error defaults to exit 1 (run all CI), never skip by mistake.
#

set -o pipefail

LOG_PREFIX="[xpu-only]"

# =============================================================================
# Emergency kill switch: globally disable CI skipping
# =============================================================================
if [[ "${DISABLE_HW_CI_SKIP:-false}" == "true" ]]; then
    echo "$LOG_PREFIX Emergency override: DISABLE_HW_CI_SKIP=true, running all CI"
    exit 1
fi

# =============================================================================
# Force full CI run: commit message contains test=all
# =============================================================================
COMMIT_MSG=$(git log -1 --pretty=%B 2>/dev/null || echo "")
if echo "$COMMIT_MSG" | grep -qiE 'test[=:] *all'; then
    echo "$LOG_PREFIX Force full CI: commit message contains test=all"
    exit 1
fi

# =============================================================================
# Collect changed files
# =============================================================================

CHANGED_FILES=""

# Method 1: GitHub PR context (diff against base branch)
if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    CHANGED_FILES=$(git diff --name-only "origin/${GITHUB_BASE_REF}...HEAD" 2>/dev/null || echo "")
fi

# Method 2: Diff against develop branch
if [[ -z "$CHANGED_FILES" ]]; then
    CHANGED_FILES=$(git diff --name-only origin/develop...HEAD 2>/dev/null || echo "")
fi

# Method 3: Diff against previous commit (fallback)
if [[ -z "$CHANGED_FILES" ]]; then
    CHANGED_FILES=$(git diff --name-only HEAD~1 2>/dev/null || echo "")
fi

# Cannot detect changed files -> run all CI
if [[ -z "$CHANGED_FILES" ]]; then
    echo "$LOG_PREFIX WARN: No changed files detected, defaulting to full CI"
    exit 1
fi

TOTAL_FILES=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
echo "$LOG_PREFIX Analyzing $TOTAL_FILES changed files"

# =============================================================================
# XPU-specific path definitions
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
    # fluid (legacy framework)
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
# Check each file: verify all belong to XPU-specific paths
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
