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

source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils.sh
init

if [ "$4" == "sot" ]; then
    PY_VERSION=$5
    ln -sf $(which python${PY_VERSION}) /usr/local/bin/python
    ln -sf $(which pip${PY_VERSION}) /usr/local/bin/pip
fi

if [ "$4" == "py3" ]; then
    PATH=/usr/local/bin:${PATH}
    ln -sf $(which python3.9) /usr/local/bin/python
    ln -sf $(which pip3.9) /usr/local/bin/pip
    pip config set global.cache-dir "/home/data/cfs/.cache/pip"

    echo "::group::Installing python dependencies"
    pip install -r "${work_dir}/python/requirements.txt"
    pip install -r "${work_dir}/python/unittest_py/requirements.txt"
    echo "::endgroup::"
fi

if [ "$4" == "coverage" ]; then
    ln -sf $(which python3.9) /usr/local/bin/python
    ln -sf $(which pip3.9) /usr/local/bin/pip
    apt install zstd -y
    pip config set global.cache-dir "/root/.cache/pip"
    pip install --upgrade pip
    echo "::group::Install python dependencies"
    pip install -r "${work_dir}/python/requirements.txt"
    pip install -r "${work_dir}/python/unittest_py/requirements.txt"
    echo "::endgroup::"
fi

function run_setup(){
    startTime_s=`date +%s`
    mkdir -p ${PADDLE_ROOT}/build
    cd ${PADDLE_ROOT}/build
    # Build script will not fail if *.deb does not exist
    rm *.deb 2>/dev/null || true
    # Delete previous built egg packages
    rm -rf ${PADDLE_ROOT}/dist 2>/dev/null || true
    # Delete previous built paddle cache
    rm -rf ${PADDLE_ROOT}/build/python/paddle 2>/dev/null || true

    SYSTEM=`uname -s`
    if [ "$SYSTEM" == "Darwin" ]; then
        echo "Using python abi: $1"
        if [ "$1" == "cp38-cp38" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.8" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.8/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.8/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.8/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.8/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.8/include/python3.8/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.8/lib/libpython3.8.dylib
                pip3.8 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp39-cp39" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.9" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.9/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.9/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.9/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.9/include/python3.9/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.9/lib/libpython3.9.dylib
                pip3.9 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp310-cp310" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.10" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.10/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.10/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.10/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.10/lib/libpython3.10.dylib
                pip3.10 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp311-cp311" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.11" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.11/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.11/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.11/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.11/include/python3.11/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib
                pip3.11 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        elif [ "$1" == "cp312-cp312" ]; then
            if [ -d "/Library/Frameworks/Python.framework/Versions/3.12" ]; then
                export LD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.12/lib/
                export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/Library/Frameworks/Python.framework/Versions/3.12/lib/
                export PATH=/Library/Frameworks/Python.framework/Versions/3.12/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
                export PYTHON_INCLUDE_DIR=/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12/
                export PYTHON_LIBRARY=/Library/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib
                pip3.12 install --user -r ${PADDLE_ROOT}/python/requirements.txt
            else
                exit 1
            fi
        fi
    else
        if [ "$1" != "" ]; then
            echo "using python abi: $1"
            if [ "$1" == "cp38-cp38" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/opt/_internal/cpython-3.8.0/bin/python3.8
                export PYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.8.0/include/python3.8
                export PYTHON_LIBRARIES=/opt/_internal/cpython-3.8.0/lib/libpython3.so
                pip3.8 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.8 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
            elif [ "$1" == "cp39-cp39" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.9.0/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/opt/_internal/cpython-3.9.0/bin/python3.9
                export PYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.9.0/include/python3.9
                export PYTHON_LIBRARIES=/opt/_internal/cpython-3.9.0/lib/libpython3.so
                pip3.9 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.9 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
            elif [ "$1" == "cp310-cp310" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.10.0/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/opt/_internal/cpython-3.10.0/bin/python3.10
                export PYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.10.0/include/python3.10
                export PYTHON_LIBRARIES=/opt/_internal/cpython-3.10.0/lib/libpython3.so
                pip3.10 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.10 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
            elif [ "$1" == "cp311-cp311" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.11.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.11.0/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/opt/_internal/cpython-3.11.0/bin/python3.11
                export PYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.11.0/include/python3.11
                export PYTHON_LIBRARIES=/opt/_internal/cpython-3.11.0/lib/libpython3.so
                pip3.11 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.11 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
            elif [ "$1" == "cp312-cp312" ]; then
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.12.0/lib/:${LD_LIBRARY_PATH}
                export PATH=/opt/_internal/cpython-3.12.0/bin/:${PATH}
                #after changing "PYTHON_LIBRARY:FILEPATH" to "PYTHON_LIBRARY" ,we can use export
                export PYTHON_EXECUTABLE=/opt/_internal/cpython-3.12.0/bin/python3.12
                export PYTHON_INCLUDE_DIR=/opt/_internal/cpython-3.12.0/include/python3.12
                export PYTHON_LIBRARIES=/opt/_internal/cpython-3.12.0/lib/libpython3.so
                pip3.12 install -r ${PADDLE_ROOT}/python/requirements.txt
                pip3.12 install -r ${PADDLE_ROOT}/paddle/scripts/compile_requirements.txt
           fi
        else
            echo "::group::Installing python dependencies"
            pip install -r ${PADDLE_ROOT}/python/requirements.txt
            echo "::endgroup::"
        fi
    fi

    if [ "$SYSTEM" == "Darwin" ]; then
        WITH_DISTRIBUTE="OFF"
        WITH_AVX=${WITH_AVX:-ON}
        WITH_ARM=${WITH_ARM:-OFF}
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-~/.cache/inference_demo}
    else
        INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR:-/root/.cache/inference_demo}
    fi

    if [ -z "${WITH_CPP_TEST}" ] && [ "${WITH_TESTING}" == "ON" ];then
      echo "::group::Installing PyGithub"
      pip install PyGithub
      echo "::endgroup::"
      python ${PADDLE_ROOT}/tools/check_only_change_python_files.py
      if [ -f "${PADDLE_ROOT}/build/only_change_python_file.txt" ];then
          export WITH_CPP_TEST=OFF
      else
          export WITH_CPP_TEST=ON
      fi
    fi
    distributed_flag=${WITH_DISTRIBUTE:-OFF}
    gloo_flag=${distributed_flag}
    pscore_flag=${distributed_flag}
    pslib_flag=${WITH_PSLIB:-OFF}
    if [ "${pslib_flag}" == "ON" ];then
      pscore_flag=${WITH_PSCORE:-OFF}
    fi

    if [ "$CMD" != "assert_file_approvals" ];then
      which python
      python -V
      python -m pip install distro
      python ${PADDLE_ROOT}/tools/summary_env.py
      bash ${PADDLE_ROOT}/tools/get_cpu_info.sh
    fi
    echo "if you use setup.py to compile,please export envs as following in /paddle ..."
    cat << EOF
    ========================================
    export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} WITH_GPU=${WITH_GPU:-OFF} WITH_SHARED_PHI=${WITH_SHARED_PHI:-OFF} WITH_TENSORRT=${WITH_TENSORRT:-ON} WITH_OPENVINO=${WITH_OPENVINO:-OFF} WITH_ROCM=${WITH_ROCM:-OFF} WITH_CINN=${WITH_CINN:-OFF} WITH_DISTRIBUTE=${distributed_flag} WITH_MKL=${WITH_MKL:-ON} WITH_AVX=${WITH_AVX:-OFF} CUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} NEW_RELEASE_PYPI=${NEW_RELEASE_PYPI:-OFF} NEW_RELEASE_ALL=${NEW_RELEASE_ALL:-OFF} NEW_RELEASE_JIT=${NEW_RELEASE_JIT:-OFF} WITH_PYTHON=${WITH_PYTHON:-ON} CUDNN_ROOT=/usr/ WITH_TESTING=${WITH_TESTING:-ON} WITH_COVERAGE=${WITH_COVERAGE:-OFF} WITH_INCREMENTAL_COVERAGE=${WITH_INCREMENTAL_COVERAGE:-OFF} CMAKE_MODULE_PATH=/opt/rocm/hip/cmake CMAKE_EXPORT_COMPILE_COMMANDS=ON WITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON} INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR} PY_VERSION=${PY_VERSION:-3.8} CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build} WITH_PSCORE=${pscore_flag} WITH_PSLIB=${pslib_flag} WITH_GLOO=${gloo_flag} WITH_XPU=${WITH_XPU:-OFF} WITH_IPU=${WITH_IPU:-OFF} XPU_SDK_ROOT=${XPU_SDK_ROOT:-""} WITH_XPU_BKCL=${WITH_XPU_BKCL:-OFF} WITH_XPU_XRE5=${WITH_XPU_XRE5:-OFF} WITH_ARM=${WITH_ARM:-OFF} WITH_STRIP=${WITH_STRIP:-ON} ON_INFER=${ON_INFER:-OFF} WITH_HETERPS=${WITH_HETERPS:-OFF} CUDA_ARCH_BIN=${CUDA_ARCH_BIN} WITH_RECORD_BUILDTIME=${WITH_RECORD_BUILDTIME:-OFF} WITH_UNITY_BUILD=${WITH_UNITY_BUILD:-OFF} WITH_ONNXRUNTIME=${WITH_ONNXRUNTIME:-OFF} WITH_CUDNN_FRONTEND=${WITH_CUDNN_FRONTEND:-OFF} WITH_CPP_TEST=${WITH_CPP_TEST:-OFF} FA_BUILD_WITH_CACHE=${FA_BUILD_WITH_CACHE:-ON}
    ========================================
EOF
    echo "if you use cmake to compile,please Configuring cmake in /paddle/build ..."
    cat <<EOF
    ========================================
    cmake .. -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} -DWITH_GPU=${WITH_GPU:-OFF} -DWITH_SHARED_PHI=${WITH_SHARED_PHI:-OFF} -DWITH_TENSORRT=${WITH_TENSORRT:-ON} -DWITH_OPENVINO=${WITH_OPENVINO:-OFF} -DWITH_ROCM=${WITH_ROCM:-OFF} -DWITH_CINN=${WITH_CINN:-OFF} -DWITH_DISTRIBUTE=${distributed_flag} -DWITH_MKL=${WITH_MKL:-ON} -DWITH_AVX=${WITH_AVX:-OFF} -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All} -DNEW_RELEASE_PYPI=${NEW_RELEASE_PYPI:-OFF} -DNEW_RELEASE_ALL=${NEW_RELEASE_ALL:-OFF} -DNEW_RELEASE_JIT=${NEW_RELEASE_JIT:-OFF} -DWITH_PYTHON=${WITH_PYTHON:-ON} -DCUDNN_ROOT=/usr/ -DWITH_TESTING=${WITH_TESTING:-ON} -DWITH_COVERAGE=${WITH_COVERAGE:-OFF} -DWITH_INCREMENTAL_COVERAGE=${WITH_INCREMENTAL_COVERAGE:-OFF} -DCMAKE_MODULE_PATH=/opt/rocm/hip/cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DWITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON} -DINFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR} -DPY_VERSION=${PY_VERSION:-3.8} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build} -DWITH_PSCORE=${pscore_flag} -DWITH_PSLIB=${pslib_flag} -DWITH_GLOO=${gloo_flag} -DWITH_XPU=${WITH_XPU:-OFF} -DWITH_IPU=${WITH_IPU:-OFF} -DXPU_SDK_ROOT=${XPU_SDK_ROOT:-""} -DWITH_XPU_BKCL=${WITH_XPU_BKCL:-OFF} -DWITH_XPU_XRE5=${WITH_XPU_XRE5:-OFF} -DWITH_ARM=${WITH_ARM:-OFF} -DWITH_STRIP=${WITH_STRIP:-ON} -DON_INFER=${ON_INFER:-OFF} -DWITH_HETERPS=${WITH_HETERPS:-OFF} -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN} -DWITH_RECORD_BUILDTIME=${WITH_RECORD_BUILDTIME:-OFF} -DWITH_UNITY_BUILD=${WITH_UNITY_BUILD:-OFF} -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME:-OFF} -DWITH_CUDNN_FRONTEND=${WITH_CUDNN_FRONTEND:-OFF} -DWITH_CPP_TEST=${WITH_CPP_TEST:-OFF} -DFA_BUILD_WITH_CACHE=${FA_BUILD_WITH_CACHE:-ON}
    ========================================
EOF
    export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
    export WITH_GPU=${WITH_GPU:-OFF}
    export WITH_TENSORRT=${WITH_TENSORRT:-ON}
    export WITH_OPENVINO=${WITH_OPENVINO:-OFF}
    export WITH_ROCM=${WITH_ROCM:-OFF}
    export WITH_CINN=${WITH_CINN:-OFF}
    export WITH_DISTRIBUTE=${distributed_flag}
    export WITH_MKL=${WITH_MKL:-ON}
    export WITH_AVX=${WITH_AVX:-OFF}
    export CUDA_ARCH_NAME=${CUDA_ARCH_NAME:-All}
    export NEW_RELEASE_PYPI=${NEW_RELEASE_PYPI:-OFF}
    export NEW_RELEASE_ALL=${NEW_RELEASE_ALL:-OFF}
    export NEW_RELEASE_JIT=${NEW_RELEASE_JIT:-OFF}
    export WITH_PYTHON=${WITH_PYTHON:-ON}
    export CUDNN_ROOT=/usr/
    export WITH_TESTING=${WITH_TESTING:-ON}
    export WITH_COVERAGE=${WITH_COVERAGE:-OFF}
    export WITH_INCREMENTAL_COVERAGE=${WITH_INCREMENTAL_COVERAGE:-OFF}
    export CMAKE_MODULE_PATH=/opt/rocm/hip/cmake
    export CMAKE_EXPORT_COMPILE_COMMANDS=ON
    export WITH_INFERENCE_API_TEST=${WITH_INFERENCE_API_TEST:-ON}
    export INFERENCE_DEMO_INSTALL_DIR=${INFERENCE_DEMO_INSTALL_DIR}
    export PY_VERSION=${PY_VERSION:-3.8}
    export CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX:-/paddle/build}
    export WITH_PSCORE=${pscore_flag}
    export WITH_PSLIB=${pslib_flag}
    export WITH_GLOO=${gloo_flag}
    export WITH_XPU=${WITH_XPU:-OFF}
    export WITH_IPU=${WITH_IPU:-OFF}
    export XPU_SDK_ROOT=${XPU_SDK_ROOT:-""}
    export WITH_XPU_BKCL=${WITH_XPU_BKCL:-OFF}
    export WITH_XPU_XRE5=${WITH_XPU_XRE5:-OFF}
    export WITH_ARM=${WITH_ARM:-OFF}
    export WITH_STRIP=${WITH_STRIP:-ON}
    export ON_INFER=${ON_INFER:-OFF}
    export WITH_HETERPS=${WITH_HETERPS:-OFF}
    export CUDA_ARCH_BIN=${CUDA_ARCH_BIN}
    export WITH_RECORD_BUILDTIME=${WITH_RECORD_BUILDTIME:-OFF}
    export WITH_UNITY_BUILD=${WITH_UNITY_BUILD:-OFF}
    export WITH_ONNXRUNTIME=${WITH_ONNXRUNTIME:-OFF}
    export WITH_CUDNN_FRONTEND=${WITH_CUDNN_FRONTEND:-OFF}
    export WITH_SHARED_PHI=${WITH_SHARED_PHI:-OFF}
    export WITH_NVCC_LAZY=${WITH_NVCC_LAZY:-ON}
    export WITH_CPP_TEST=${WITH_CPP_TEST:-ON}
    export FA_BUILD_WITH_CACHE=${FA_BUILD_WITH_CACHE:-ON}


    if [ "$SYSTEM" == "Linux" ];then
      if [ `nproc` -gt 16 ];then
          parallel_number=$(expr `nproc` - 8)
      else
          parallel_number=`nproc`
      fi
    else
      parallel_number=8
    fi
    if [ "$3" != "" ]; then
      parallel_number=$3
    fi

    # reset ccache zero stats for collect PR's actual hit rate
    if [ "${MAX_JOBS}" == "" ]; then
        export MAX_JOBS=${parallel_number}
    fi
    ccache -z
    cd ..
    echo "::group::Build Paddle"
    if [ "${PYTHON_EXECUTABLE}" != "" ];then
        if [ "$SYSTEM" == "Darwin" ]; then
            ${PYTHON_EXECUTABLE} setup.py $2 --plat-name=macosx_10_9_x86_64;build_error=$?
        else
            ${PYTHON_EXECUTABLE} setup.py $2;build_error=$?
        fi
    else
        if [ "$SYSTEM" == "Darwin" ]; then
            python setup.py $2 --plat-name=macosx_10_9_x86_64;build_error=$?
        else
            python setup.py $2;build_error=$?
        fi
    fi
    echo "::endgroup::"

    # ci will collect ccache hit rate
    collect_ccache_hits


    if [ "$build_error" != 0 ];then
        exit 7;
    fi

    build_size "" $6

    endTime_s=`date +%s`
    [ -n "$startTime_firstBuild" ] && startTime_s=$startTime_firstBuild
    echo "Build Time: $[ $endTime_s - $startTime_s ]s"
    echo "ipipe_log_param_Build_Time: $[ $endTime_s - $startTime_s ]s" >> ${PADDLE_ROOT}/build/build_summary.txt
}

run_setup "$@"

if [[ -f ${PADDLE_ROOT}/build/build_summary.txt ]];then
echo "=====================build summary======================"
cat ${PADDLE_ROOT}/build/build_summary.txt
echo "========================================================"
fi
