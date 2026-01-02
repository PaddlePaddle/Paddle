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

limit_ccache_size() {
    echo "Setting ccache max size to 15G"
    ccache --max-size=15G
}

if [ "$MODE" == "restore" ]; then
    echo "Starting ccache restore..."
    echo "Ccache version:"
    ccache --version
    echo "CFS Cache Path: ${CFS_CACHE_PATH}"
    echo "Local Cache Path: ${LOCAL_CACHE_PATH}"
    mkdir -p "${LOCAL_CACHE_PATH}"
    if [ -d "${CFS_CACHE_PATH}" ]; then
        echo "::group::Restoring ccache from CFS..."
        limit_ccache_size
        # echo "CFS cache size:"
        # du -sh "${CFS_CACHE_PATH}" || echo "Failed to get size of CFS cache."
        if command -v ccache &> /dev/null; then
            echo "Cleaning unused files in CFS ccache before transfer..."
            ccache --dir "${CFS_CACHE_PATH}" --clean || echo "ccache cleanup on CFS failed"
            echo "CFS ccache stats:"
            ccache --dir "${CFS_CACHE_PATH}" --show-stats || true
            echo "Local ccache stats:"
            ccache --dir "${LOCAL_CACHE_PATH}" --show-stats || true
        else
            echo "ccache not found, skipping ccache cleanup and stats."
        fi
        install_rsync
        # -a: archive mode, preserves permissions/times
        # --info=progress2: show progress (optional, might be noisy in CI)
        rsync -avzW \
            --no-perms --no-owner --no-group \
            --partial \
            --progress \
            --delete "${CFS_CACHE_PATH}/" "${LOCAL_CACHE_PATH}/"
        echo "::endgroup::"
    else
        echo "CFS cache path not found: ${CFS_CACHE_PATH}. Skipping restore."
    fi
elif [ "$MODE" == "save" ]; then
    if [ -d "$(dirname "${CFS_CACHE_PATH}")" ]; then
        echo "::group::Saving ccache to CFS..."
        install_rsync
        mkdir -p "${CFS_CACHE_PATH}"
        if command -v ccache &> /dev/null; then
            echo "Cleaning unused files in CFS ccache before transfer..."
            ccache --dir "${CFS_CACHE_PATH}" --clean || echo "ccache cleanup on CFS failed"
            echo "CFS ccache stats:"
            ccache --dir "${CFS_CACHE_PATH}" --show-stats || true
            echo "Local ccache stats:"
            ccache --dir "${LOCAL_CACHE_PATH}" --show-stats || true
        else
            echo "ccache not found, skipping ccache cleanup and stats."
        fi

        # Sync back to CFS
        rsync -avzW \
            --no-perms --no-owner --no-group \
            --partial \
            --progress \
            --delete "${LOCAL_CACHE_PATH}/" "${CFS_CACHE_PATH}/"
        echo "::endgroup::"
    else
        echo "CFS parent directory not found. Skipping cache save."
    fi
else
    echo "Usage: $0 [restore|save]"
    exit 1
fi
