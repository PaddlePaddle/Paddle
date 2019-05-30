#!/bin/bash
set -ex

TESTS_FILE="./lite_tests.txt"

readonly common_flags="-DWITH_LITE=ON -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF -DWITH_PYTHON=OFF -DWITH_TESTING=ON -DLITE_WITH_ARM=OFF"
function cmake_x86 {
    cmake ..  -DWITH_GPU=OFF -DWITH_MKLDNN=OFF -DLITE_WITH_X86=ON ${common_flags}
}

function cmake_gpu {
    cmake .. " -DWITH_GPU=ON {common_flags} -DLITE_WITH_GPU=ON"
}

function cmake_arm {
    # ARM_TARGET_OS="android" , "armlinux"
    # ARM_TARGET_ARCH_ABI = "arm64-v8a", "armeabi-v7a" ,"armeabi-v7a-hf"
    for os in "android" "armlinux" ; do
        for abi in "arm64-v8a" "armeabi-v7a" "armeabi-v7a-hf" ; do
            if [[ ${abi} == "armeabi-v7a-hf" ]]; then
                echo "armeabi-v7a-hf is not supported on both android and armlinux"
                continue
            fi

            if [[ ${os} == "armlinux" && ${abi} == "armeabi-v7a" ]]; then
                echo "armeabi-v7a is not supported on armlinux yet"
                continue
            fi

            echo "Build for ${os} ${abi}"

            build_dir=build.lite.${os}.${abi}
            mkdir -p $build_dir
            cd $build_dir

            cmake .. \
                -DWITH_GPU=OFF \
                -DWITH_LITE=ON \
                -DLITE_WITH_CUDA=OFF \
                -DLITE_WITH_X86=OFF \
                -DLITE_WITH_ARM=ON \
                -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
                -DWITH_TESTING=ON \
                -DWITH_MKL=OFF \
                -DARM_TARGET_OS=${os} -DARM_TARGET_ARCH_ABI=${abi} 

            make test_fc_compute_arm -j
            make test_softmax_compute_arm -j
            make cxx_api_lite_bin -j
            cd -

        done
    done

}

function build {
    file=$1
    for _test in $(cat $file); do
        make $_test -j8
    done
}

# It will eagerly test all lite related unittests.
function test_lite {
    local file=$1
    echo "file: ${file}"
    for _test in $(cat $file); do
        ctest -R $_test -V
    done
}

# Run test on mobile
function test_mobile {
    # TODO(XXX) Implement this
    local file=$1
}

# Build the code and run lite server tests. This is executed in the CI system.
function build_test_server {
    mkdir -p ./build
    cd ./build
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/paddle/build/third_party/install/mklml/lib"
    cmake_x86
    build $TESTS_FILE
    test_lite $TESTS_FILE
}

############################# MAIN #################################
function print_usage {
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "cmake_x86: run cmake with X86 mode"
    echo -e "cmake_cuda: run cmake with CUDA mode"
    echo -e "cmake_arm: run cmake with ARM mode"
    echo
    echo -e "build: compile the tests"
    echo
    echo -e "test_server: run server tests"
    echo -e "test_mobile: run mobile tests"
    echo "----------------------------------------"
    echo
}

function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            --tests=*)
                TESTS_FILE="${i#*=}"
                shift
                ;;
            build)
                build $TESTS_FILE
                shift
                ;;
            cmake_x86)
                cmake_x86
                shift
                ;;
            cmake_cuda)
                cmake_cuda
                shift
                ;;
            cmake_arm)
                cmake_arm
                shift
                ;;
            test_server)
                test_lite $TESTS_FILE
                shift
                ;;
            test_mobile)
                test_lite $TESTS_FILE
                shift
                ;;
            build_test_server)
                build_test_server
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
}

print_usage

main $@
