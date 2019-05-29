#!/bin/bash
set -ex

TESTS_FILE="./lite_tests.txt"

readonly common_flags="-DWITH_LITE=ON -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF -DWITH_PYTHON=OFF -DWITH_TESTING=ON -DLITE_WITH_ARM=OFF"
function cmake_x86 {
    cmake ..  -DWITH_GPU=OFF -DLITE_WITH_X86=ON ${common_flags}
}

function cmake_gpu {
    cmake .. " -DWITH_GPU=ON {common_flags} -DLITE_WITH_GPU=ON"
}

function cmake_arm {
    cmake .. \
          -DWITH_GPU=OFF \
          -DWITH_LITE=ON \
          -DLITE_WITH_X86=OFF \
          -DLITE_WITH_CUDA=OFF \
          -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
          -DWITH_TESTING=ON \
          -DWITH_MKL=OFF \
          -DWITH_MKLDNN=OFF
    make cxx_api_lite_bin -j8
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

# Build the code and run lite server tests.
function build_test_server {
    cd ./build
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
