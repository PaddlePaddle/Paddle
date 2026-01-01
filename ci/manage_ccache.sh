#!/bin/bash

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

set -e

MODE=$1
CFS_CACHE_PATH=${CFS_CCACHE_DIR}
LOCAL_CACHE_PATH=${CCACHE_DIR}

if [ -z "${CFS_CACHE_PATH}" ] || [ -z "${LOCAL_CACHE_PATH}" ]; then
    echo "Error: CFS_CCACHE_DIR and CCACHE_DIR environment variables must be set."
    exit 1
fi

install_rsync() {
    if ! command -v rsync &> /dev/null; then
        echo "Installing rsync..."
        apt-get update && apt-get install -y rsync
    fi
}

if [ "$MODE" == "restore" ]; then
    mkdir -p "${LOCAL_CACHE_PATH}"
    if [ -d "${CFS_CACHE_PATH}" ]; then
        echo "::group::Restoring ccache from CFS..."
        install_rsync
        # -a: archive mode, preserves permissions/times
        # --info=progress2: show progress (optional, might be noisy in CI)
        rsync -a "${CFS_CACHE_PATH}/" "${LOCAL_CACHE_PATH}/"
        echo "::endgroup::"
    else
        echo "CFS cache path not found: ${CFS_CACHE_PATH}. Skipping restore."
    fi
elif [ "$MODE" == "save" ]; then
    if [ -d "$(dirname "${CFS_CACHE_PATH}")" ]; then
        echo "::group::Saving ccache to CFS..."
        install_rsync
        mkdir -p "${CFS_CACHE_PATH}"
        # Sync back to CFS
        rsync -a "${LOCAL_CACHE_PATH}/" "${CFS_CACHE_PATH}/"
        echo "::endgroup::"
    else
        echo "CFS parent directory not found. Skipping cache save."
    fi
else
    echo "Usage: $0 [restore|save]"
    exit 1
fi
