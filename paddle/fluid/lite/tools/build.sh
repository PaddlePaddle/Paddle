#!/bin/bash
set -ex

TESTS_FILE="./lite_tests.txt"
LIBS_FILE="./lite_libs.txt"

readonly common_flags="-DWITH_LITE=ON -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF -DWITH_PYTHON=OFF -DWITH_TESTING=ON -DLITE_WITH_ARM=OFF"

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_for_codegen {
    # in build directory
    mkdir -p ./paddle/fluid/lite/gen_code
    touch ./paddle/fluid/lite/gen_code/__generated_code__.cc
}
function cmake_x86 {
    prepare_for_codegen
    cmake ..  -DWITH_GPU=OFF -DWITH_MKLDNN=OFF -DLITE_WITH_X86=ON ${common_flags}
}

function cmake_x86_for_CI {
    prepare_for_codegen
    cmake ..  -DWITH_GPU=OFF -DWITH_MKLDNN=OFF -DLITE_WITH_X86=ON ${common_flags} -DLITE_WITH_PROFILE=ON
}

function cmake_gpu {
    prepare_for_codegen
    cmake .. " -DWITH_GPU=ON {common_flags} -DLITE_WITH_GPU=ON"
}

function check_style {
    pip install cpplint
    export PATH=/usr/bin:$PATH
    pre-commit install
    clang-format --version

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi
}

function cmake_arm {
    # $1: ARM_TARGET_OS in "android" , "armlinux"
    # $2: ARM_TARGET_ARCH_ABI in "arm64-v8a", "armeabi-v7a" ,"armeabi-v7a-hf"
    cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=ON \
        -DARM_TARGET_OS=$1 -DARM_TARGET_ARCH_ABI=$2
}

function build {
    file=$1
    for _test in $(cat $file); do
        #make $_test -j$(expr $(nproc) - 2)
        make $_test -j8
    done
}

# It will eagerly test all lite related unittests.
function test_lite {
    local file=$1
    echo "file: ${file}"

    for _test in $(cat $file); do
        # We move the build phase here to make the 'gen_code' test compiles after the
        # corresponding test is executed and the C++ code generates.
        make $_test -j$(expr $(nproc) - 2)
        ctest -R $_test -V
    done
}

port_armv8=5554
port_armv7=5556

# Run test on android
function test_lite_android {
    local file=$1
    local adb_abi=$2
    local port=
    if [[ ${adb_abi} == "armeabi-v7a" ]]; then
        port=${port_armv7}
    fi

    if [[ ${adb_abi} == "arm64-v8a" ]]; then
        port=${port_armv8}
    fi
    if [[ "${port}x" == "x" ]]; then
        echo "Port can not be empty"
        exit 1
    fi

    echo "file: ${file}"
    # push all to adb and test
    adb_work_dir="/data/local/tmp"
    skip_list="test_model_parser_lite"
    for _test in $(cat $file); do
        [[ $skip_list =~ (^|[[:space:]])$_test($|[[:space:]]) ]] && continue || echo 'skip $_test'
        testpath=$(find ./paddle/fluid -name ${_test})
        adb -s emulator-${port} push ${testpath} ${adb_work_dir}
        adb -s emulator-${port} shell chmod +x "${adb_work_dir}/${_test}"
        adb -s emulator-${port} shell "./${adb_work_dir}/${_test}"
    done
}

# Build the code and run lite server tests. This is executed in the CI system.
function build_test_server {
    rm -rf build
    mkdir -p ./build
    cd ./build
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/paddle/build/third_party/install/mklml/lib"
    cmake_x86_for_CI
    # compile the tests and execute them.
    test_lite $TESTS_FILE
    # build the remaining libraries to check compiling error.
    build $LIBS_FILE
}

# Build the code and run lite server tests. This is executed in the CI system.
function build_test_arm {
    adb kill-server
    adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
    # start android arm64-v8a armeabi-v7a emulators first
    echo n | avdmanager create avd -f -n paddle-armv8 -k "system-images;android-24;google_apis;arm64-v8a"
    echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv8 -noaudio -no-window -gpu off -verbose -port ${port_armv8} &
    sleep 1m
    echo n | avdmanager create avd -f -n paddle-armv7 -k "system-images;android-24;google_apis;armeabi-v7a"
    echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv7 -noaudio -no-window -gpu off -verbose -port ${port_armv7} &
    sleep 1m

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

            build_dir=build.lite.${os}.${abi}
            mkdir -p $build_dir
            cd $build_dir
            cmake_arm ${os} ${abi}
            build $TESTS_FILE

            if [[ ${os} == "android" ]]; then
                adb_abi=${abi}
                if [[ ${adb_abi} == "armeabi-v7a-hf" ]]; then
                    adb_abi="armeabi-v7a"
                fi
                if [[ ${adb_abi} == "armeabi-v7a" ]]; then
                    # skip v7 tests
                    continue
                fi
                test_lite_android $TESTS_FILE ${adb_abi}
                # armlinux need in another docker
            fi
            cd -
        done
    done
    adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
    echo "Done"
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
                build $LIBS_FILE
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
                cmake_arm $2 $3
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
            build_test_arm)
                build_test_arm
                shift
                ;;
            check_style)
                check_style
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
