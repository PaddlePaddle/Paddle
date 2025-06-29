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

mkdir run_env
ln -s $(which python3.10) run_env/python
ln -s $(which pip3.10) run_env/pip
export PATH=$(pwd)/run_env:${PATH}
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "::group::Installing python dependencies"
pip install -r "${work_dir}/python/requirements.txt"
pip install -r "${work_dir}/python/unittest_py/requirements.txt"
echo "::endgroup::"

sed -i 's#set(cuda_arch_bin "60 61")#set(cuda_arch_bin "61")#g' cmake/cuda.cmake

python ${PADDLE_ROOT}/tools/remove_grad_op_and_kernel.py

source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils.sh
init

function gen_fluid_lib_by_setup(){
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    cat <<EOF
    ========================================
    Generating fluid library for train and inference ...
    ========================================
EOF
    parallel_number=`nproc`
    if [[ "$1" != "" ]]; then
      parallel_number=$1
    fi
    startTime_s=`date +%s`
    set +e
    export MAX_JOBS=${parallel_number}
    export WITH_DISTRIBUTE=OFF ON_INFER=ON WITH_TENSORRT=ON CUDA_ARCH_NAME=${CUDA_ARCH_NAME:-Auto} WITH_PYTHON=${WITH_PYTHON:-ON} WITH_ONNXRUNTIME=${WITH_ONNXRUNTIME:-OFF} WITH_SHARED_PHI=${WITH_SHARED_PHI:-ON}
    echo "if you use setup.py to compile,please export envs as following in /paddle ..."
    cat << EOF
    ========================================
    export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} WITH_GPU=${WITH_GPU:-OFF} WITH_SHARED_PHI=${WITH_SHARED_PHI:-ON} WITH_TENSORRT=${WITH_TENSORRT:-ON} WITH_ROCM=${WITH_ROCM:-OFF} WITH_CINN=${WITH_CINN:-OFF} WITH_DISTRIBUTE=${distributed_flag} WITH_MKL=${WITH_MKL:-ON} WITH_AVX=${WITH_AVX:-OFF} CUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} NEW_RELEASE_PYPI=${NEW_RELEASE_PYPI:-OFF} NEW_RELEASE_ALL=${NEW_RELEASE_ALL:-OFF} NEW_RELEASE_JIT=${NEW_RELEASE_JIT:-OFF} WITH_PYTHON=${WITH_PYTHON:-ON} CUDNN_ROOT=/usr/ WITH_TESTING=${WITH_TESTING:-ON} WITH_COVERAGE=${WITH_COVERAGE:-OFF} WITH_INCREMENTAL_COVERAGE=${WITH_INCREMENTAL_COVERAGE:-OFF} CMAKE_MODULE_PATH=/opt/rocm/hip/cmake CMAKE_EXPORT_COMPILE_COMMANDS=ON WITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON} INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR} PY_VERSION=${PY_VERSION:-3.8} CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build} WITH_PSCORE=${pscore_flag} WITH_PSLIB=${pslib_flag} WITH_GLOO=${gloo_flag} WITH_XPU=${WITH_XPU:-OFF} WITH_IPU=${WITH_IPU:-OFF} XPU_SDK_ROOT=${XPU_SDK_ROOT:-""} WITH_XPU_BKCL=${WITH_XPU_BKCL:-OFF} WITH_ARM=${WITH_ARM:-OFF} WITH_STRIP=${WITH_STRIP:-ON} ON_INFER=${ON_INFER:-OFF} WITH_HETERPS=${WITH_HETERPS:-OFF} CUDA_ARCH_BIN=${CUDA_ARCH_BIN} WITH_RECORD_BUILDTIME=${WITH_RECORD_BUILDTIME:-OFF} WITH_UNITY_BUILD=${WITH_UNITY_BUILD:-OFF} WITH_ONNXRUNTIME=${WITH_ONNXRUNTIME:-OFF} WITH_CUDNN_FRONTEND=${WITH_CUDNN_FRONTEND:-OFF}
    ========================================
EOF
    echo "if you use cmake to compile,please Configuring cmake in /paddle/build ..."
    cat <<EOF
    ========================================
    cmake .. -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} -DWITH_GPU=${WITH_GPU:-OFF} -DWITH_SHARED_PHI=${WITH_SHARED_PHI:-ON} -DWITH_TENSORRT=${WITH_TENSORRT:-ON} -DWITH_ROCM=${WITH_ROCM:-OFF} -DWITH_CINN=${WITH_CINN:-OFF} -DWITH_DISTRIBUTE=${distributed_flag} -DWITH_MKL=${WITH_MKL:-ON} -DWITH_AVX=${WITH_AVX:-OFF} -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} -DNEW_RELEASE_PYPI=${NEW_RELEASE_PYPI:-OFF} -DNEW_RELEASE_ALL=${NEW_RELEASE_ALL:-OFF} -DNEW_RELEASE_JIT=${NEW_RELEASE_JIT:-OFF} -DWITH_PYTHON=${WITH_PYTHON:-ON} -DCUDNN_ROOT=/usr/ -DWITH_TESTING=${WITH_TESTING:-ON} -DWITH_COVERAGE=${WITH_COVERAGE:-OFF} -DWITH_INCREMENTAL_COVERAGE=${WITH_INCREMENTAL_COVERAGE:-OFF} -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON} -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR} -DPY_VERSION=${PY_VERSION:-3.8} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build} -DWITH_PSCORE=${pscore_flag} -DWITH_PSLIB=${pslib_flag} -DWITH_GLOO=${gloo_flag} -DWITH_XPU=${WITH_XPU:-OFF} -DWITH_IPU=${WITH_IPU:-OFF} -DXPU_SDK_ROOT=${XPU_SDK_ROOT:-""} -DWITH_XPU_BKCL=${WITH_XPU_BKCL:-OFF} -DWITH_ARM=${WITH_ARM:-OFF} -DWITH_STRIP=${WITH_STRIP:-ON} -DON_INFER=${ON_INFER:-OFF} -DWITH_HETERPS=${WITH_HETERPS:-OFF} -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} -DWITH_RECORD_BUILDTIME=${WITH_RECORD_BUILDTIME:-OFF} -DWITH_UNITY_BUILD=${WITH_UNITY_BUILD:-OFF} -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME:-OFF} -DWITH_CUDNN_FRONTEND=${WITH_CUDNN_FRONTEND:-OFF}
    ========================================
EOF
    # reset ccache zero stats for collect PR's actual hit rate
    ccache -z
    cd ..
    if [ "${PYTHON_EXECUTABLE}" != "" ];then
        ${PYTHON_EXECUTABLE} setup.py bdist_wheel;build_error=$?
    else
        python setup.py bdist_wheel;build_error=$?
    fi
    # ci will collect ccache hit rate
    collect_ccache_hits

    if [ "$build_error" != 0 ];then
        exit 7;
    fi
    endTime_s=`date +%s`
    echo "Build Time: $[ $endTime_s - $startTime_s ]s"
    echo "ipipe_log_param_Build_Time: $[ $endTime_s - $startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt

    build_size "paddle_inference"
    build_size "paddle_inference_c"
}

gen_fluid_lib_by_setup ${parallel_number_env:-""}
