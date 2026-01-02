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

if [ "$MODE" == "restore" ]; then
    echo "Starting ccache restore..."
    echo "Ccache version:"
    ccache --version
    echo "CFS Cache Path: ${CFS_CACHE_PATH}"
    echo "Local Cache Path: ${LOCAL_CACHE_PATH}"
    rm -rf "${LOCAL_CACHE_PATH}"
    mkdir -p "${LOCAL_CACHE_PATH}"
    if [ -d "${CFS_CACHE_PATH}" ]; then
        echo "::group::Restoring ccache from CFS..."
        # create a tarball from CFS cache
        tar -cf /tmp/ccache_backup.tar -C "${CFS_CACHE_PATH}" .
        # extract to local ccache dir
        tar -xf /tmp/ccache_backup.tar -C "${LOCAL_CACHE_PATH}"
        echo "::endgroup::"
    else
        echo "CFS cache path not found: ${CFS_CACHE_PATH}. Skipping restore."
    fi
elif [ "$MODE" == "save" ]; then
    if [ -d "$(dirname "${CFS_CACHE_PATH}")" ]; then
        echo "::group::Saving ccache to CFS..."
        mkdir -p "${CFS_CACHE_PATH}"

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
elif [ "$MODE" == "prune" ]; then
    if [ -d "${CFS_CACHE_PATH}" ]; then
        echo "::group::Pruning ccache..."
        ccache --dir "${CFS_CACHE_PATH}" --clean || echo "ccache cleanup on CFS failed"
        echo "::endgroup::"
    else
        echo "CFS cache path not found: ${CFS_CACHE_PATH}. Skipping prune."
    fi
else
    echo "Usage: $0 [restore|save|prune]"
    exit 1
fi
