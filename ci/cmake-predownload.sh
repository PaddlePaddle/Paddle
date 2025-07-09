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

set -e
PREDOWNLOAD_DIR=${CACHE_DIR}/cmake-predownload

download_and_verify() {
    local URL="$1"
    local EXPECTED_MD5="$2"
    local TARGET_DIR=${PREDOWNLOAD_DIR}
    local TARGET_FILENAME="${3:-$(basename "${URL}")}"
    local MAX_RETRIES="${4:-3}"
    local FILE_PATH="${TARGET_DIR}/${TARGET_FILENAME}"
    local retries=0

    mkdir -p "${TARGET_DIR}" || {
        echo "Error: Failed to create directory ${TARGET_DIR}."
        return 1
    }

    while [ $retries -lt $MAX_RETRIES ]; do
        echo "Downloading ${URL} (Attempt $((retries + 1))/${MAX_RETRIES})..."
        local TEMP_FILE_PATH="${FILE_PATH}.tmp"
        if [[ "$URL" == *"github"* ]]; then
            wget -q --no-check-certificate "${URL}" -O "${TEMP_FILE_PATH}"
        else
            wget -q --no-proxy --no-check-certificate "${URL}" -O "${TEMP_FILE_PATH}"
        fi

        if [ $? -ne 0 ]; then
            echo "Download failed. Retrying..."
            rm -f "${TEMP_FILE_PATH}"
            retries=$((retries + 1))
            sleep 1
            continue
        fi

        echo "Verifying MD5 checksum..."
        local COMPUTED_MD5=$(md5sum "${TEMP_FILE_PATH}" | awk '{print $1}')

        if [ "${COMPUTED_MD5}" = "${EXPECTED_MD5}" ]; then
            mv "${TEMP_FILE_PATH}" "${FILE_PATH}"
            echo "MD5 verification succeeded. File saved to: ${FILE_PATH}"
            return 0
        else
            echo "MD5 verification failed. Expected: ${EXPECTED_MD5}, Got: ${COMPUTED_MD5}"
            rm -f "${TEMP_FILE_PATH}"
            retries=$((retries + 1))
            sleep 1
        fi
    done

    echo "Failed to download and verify after ${MAX_RETRIES} attempts."
    exit 1
}

mkdir -p ${PREDOWNLOAD_DIR}
TARGET_DIR=/paddle/third_party

# llvm.cmake
filename=llvm11-glibc2.17.tar.gz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddle-inference-dist.bj.bcebos.com/CINN/llvm11-glibc2.17.tar.gz
    EXPECTED_MD5=33c7d3cc6d370585381e8d90bd7c2198
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p /paddle/build/third_party/llvm/src
cp ${PREDOWNLOAD_DIR}/${filename} /paddle/build/third_party/llvm/src/${filename}

# isl.cmake
filename=isl-6a1760fe.tar.gz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddle-inference-dist.bj.bcebos.com/CINN/isl-6a1760fe.tar.gz
    EXPECTED_MD5=fff10083fb79d394b8a7b7b2089f6183
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p ${TARGET_DIR}/isl/
cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/isl/${filename}

# ginac.cmake
filename=ginac-1.8.1_cln-1.3.6_gmp-6.2.1.tar.gz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddle-inference-dist.bj.bcebos.com/CINN/ginac-1.8.1_cln-1.3.6_gmp-6.2.1.tar.gz
    EXPECTED_MD5=ebc3e4b7770dd604777ac3f01bfc8b06
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p ${TARGET_DIR}/ginac/
cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/ginac/${filename}

# lapack.cmake
filename=lapack_lnx_v3.10.0.20210628.tar.gz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddlepaddledeps.bj.bcebos.com/lapack_lnx_v3.10.0.20210628.tar.gz
    EXPECTED_MD5=71f8cc8237a8571692f3e07f9a4f25f6
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p ${TARGET_DIR}/lapack/Linux/
cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/lapack/Linux/${filename}

# mklml.cmake
filename=csrmm_mklml_lnx_2019.0.5.tgz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=http://paddlepaddledeps.bj.bcebos.com/csrmm_mklml_lnx_2019.0.5.tgz
    EXPECTED_MD5=bc6a7faea6a2a9ad31752386f3ae87da
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p ${TARGET_DIR}/mklml/Linux
cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/mklml/Linux/${filename}

# third_party.cmake
filename=externalErrorMsg_20210928.tar.gz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddlepaddledeps.bj.bcebos.com/externalErrorMsg_20210928.tar.gz
    EXPECTED_MD5=a712a49384e77ca216ad866712f7cafa
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p /paddle/build/third_party/externalError/data
cp ${PREDOWNLOAD_DIR}/${filename} /paddle/build/third_party/externalError/data/${filename}

# libmct.cmake
filename=libmct.tar.gz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://pslib.bj.bcebos.com/libmct/libmct.tar.gz
    EXPECTED_MD5=7e6b6c91b45b7490186f7120ef7e08fe
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p ${TARGET_DIR}/libmct/Linux
cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/libmct/Linux/${filename}

# jamalloc.cmake
filename=jemalloc-5.1.0.tar.bz2
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddle-ci.gz.bcebos.com/jemalloc-5.1.0.tar.bz2
    EXPECTED_MD5=1f47a5aff2d323c317dfa4cf23be1ce4
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p ${TARGET_DIR}/jemalloc/Linux
cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/jemalloc/Linux/${filename}

# afs_api.cmake
filename=afs_api.tar.gz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://fleet.bj.bcebos.com/heterps/afs_api.tar.gz
    EXPECTED_MD5=1fc56f4a2891f122ddb9c16ff8ca232a
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p /paddle/build/third_party/afs_api/src/extern_afs_api
cp ${PREDOWNLOAD_DIR}/${filename} /paddle/build/third_party/afs_api/src/extern_afs_api/${filename}

# dgc.cmake
filename=collective_7369ff.tgz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://fleet.bj.bcebos.com/dgc/collective_7369ff.tgz
    EXPECTED_MD5=ede459281a0f979da8d84f81287369ff
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p ${TARGET_DIR}/dgc/Linux
cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/dgc/Linux/${filename}

# cinn
filename=lite_naive_model.tar.gz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=http://paddle-inference-dist.bj.bcebos.com/lite_naive_model.tar.gz
    EXPECTED_MD5=e32ffbe183d99ff505dfecbbeef4e441
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p /paddle/build/third_party/model
cp ${PREDOWNLOAD_DIR}/${filename} /paddle/build/third_party/model/${filename}

# onnxruntime.cmake
# inference
ONNXRUNTIME_VERSION=1.11.1
SYSTEM=$(uname -s)
if [ "$SYSTEM" == "Linux" ]; then
    filename=linux-${ONNXRUNTIME_VERSION}.tgz
    echo "check ${filename}"
    if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
        echo "NO cfs cache, try to wget"
        URL=https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
        EXPECTED_MD5=ce3f2376854b3da4b483d6989666995a
        download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
    else
        echo "use cfs cache"
    fi
    mkdir -p ${TARGET_DIR}/onnxruntime/Linux
    cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/onnxruntime/Linux/${ONNXRUNTIME_VERSION}.tgz
else
    filename=osx-${ONNXRUNTIME_VERSION}.tgz
    echo "check ${filename}"
    if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
        echo "NO cfs cache, try to wget"
        URL=https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz
        EXPECTED_MD5=6a6f6b7df97587da59976042f475d3f4
        download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
    else
        echo "use cfs cache"
    fi
    mkdir -p ${TARGET_DIR}/onnxruntime/Darwin
    cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/onnxruntime/Darwin/${ONNXRUNTIME_VERSION}.tgz
fi

# paddle2onnx.cmake
# inference
PADDLE2ONNX_VERSION=1.0.0rc2
SYSTEM=$(uname -s)
if [ "$SYSTEM" == "Linux" ]; then
    filename=linux-${PADDLE2ONNX_VERSION}.tgz
    echo "check ${filename}"
    if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
        echo "NO cfs cache, try to wget"
        URL=https://github.com//PaddlePaddle/Paddle2ONNX/releases/download/v${PADDLE2ONNX_VERSION}/paddle2onnx-linux-x64-${PADDLE2ONNX_VERSION}.tgz
        EXPECTED_MD5=3fbb074987ba241327797f76514e937f
        download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
    else
        echo "use cfs cache"
    fi
    mkdir -p ${TARGET_DIR}/paddle2onnx/Linux
    cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/paddle2onnx/Linux/${PADDLE2ONNX_VERSION}.tgz
else
    filename=osx-${PADDLE2ONNX_VERSION}.tgz
    echo "check ${filename}"
    if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
        echo "NO cfs cache, try to wget"
        URL=https://github.com/PaddlePaddle/Paddle2ONNX/releases/download/v${PADDLE2ONNX_VERSION}/paddle2onnx-osx-x86_64-${PADDLE2ONNX_VERSION}.tgz
        EXPECTED_MD5=32a4381ff8441b69d58ef0fd6fd919eb
        download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
    else
        echo "use cfs cache"
    fi
    mkdir -p ${TARGET_DIR}/paddle2onnx/Darwin
    cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/paddle2onnx/Darwin/${PADDLE2ONNX_VERSION}.tgz
fi

# flashattn.cmake

# cudnn-frontend.cmake
# distribute
CUDNN_FRONTEND_VER=v1.5.2
filename=${CUDNN_FRONTEND_VER}.tar.gz
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.5.2.tar.gz
    EXPECTED_MD5=5ccb6e50a534670b2af7b30de50e7641
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p ${TARGET_DIR}/cudnn-frontend
cp ${PREDOWNLOAD_DIR}/${filename} ${TARGET_DIR}/cudnn-frontend/${filename}

# test/cpp/pir/core
filename=resnet50_main.prog
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddle-ci.gz.bcebos.com/ir_translator_test/resnet50_main.prog
    EXPECTED_MD5=b64c0ad3c96d99fc37d12094623ce1ad
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p /paddle/build/test/cpp/pir/core/
cp ${PREDOWNLOAD_DIR}/${filename} /paddle/build/test/cpp/pir/core/${filename}

filename=resnet50_startup.prog
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddle-ci.gz.bcebos.com/ir_translator_test/resnet50_startup.prog
    EXPECTED_MD5=6affc5f40f0f0bb84d956919b95eaf50
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p /paddle/build/test/cpp/pir/core/
cp ${PREDOWNLOAD_DIR}/${filename} /paddle/build/test/cpp/pir/core/${filename}

filename=conditional_block_test.prog
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddle-ci.gz.bcebos.com/ir_translator_test/conditional_block_test.prog
    EXPECTED_MD5=cf9dc869ca7f69e2d57b38dbf8427134
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p /paddle/build/test/cpp/pir/core/
cp ${PREDOWNLOAD_DIR}/${filename} /paddle/build/test/cpp/pir/core/${filename}

filename=while_op_test.prog
echo "check ${filename}"
if [ ! -f "${PREDOWNLOAD_DIR}/${filename}" ]; then
    echo "NO cfs cache, try to wget"
    URL=https://paddle-ci.gz.bcebos.com/ir_translator_test/while_op_test.prog
    EXPECTED_MD5=290164ae52a496332b0be5829fc93bcd
    download_and_verify ${URL} ${EXPECTED_MD5} ${filename}
else
    echo "use cfs cache"
fi
mkdir -p /paddle/build/test/cpp/pir/core/
cp ${PREDOWNLOAD_DIR}/${filename} /paddle/build/test/cpp/pir/core/${filename}
