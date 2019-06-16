#!/bin/bash
set -ex

TESTS_FILE="./lite_tests.txt"
LIBS_FILE="./lite_libs.txt"

readonly common_flags="-DWITH_LITE=ON -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF -DWITH_PYTHON=OFF -DWITH_TESTING=ON -DLITE_WITH_ARM=OFF"

NUM_CORES_FOR_COMPILE=8

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_for_codegen {
    # in build directory
    mkdir -p ./paddle/fluid/lite/gen_code
    touch ./paddle/fluid/lite/gen_code/__generated_code__.cc
}

function check_need_ci {
    git log -1 --oneline | grep "test=develop" || exit -1
}

function cmake_x86 {
    prepare_for_codegen
    cmake ..  -DWITH_GPU=OFF -DWITH_MKLDNN=OFF -DLITE_WITH_X86=ON ${common_flags}
}

# This method is only called in CI.
function cmake_x86_for_CI {
    prepare_for_codegen # fake an empty __generated_code__.cc to pass cmake.
    cmake ..  -DWITH_GPU=OFF -DWITH_MKLDNN=OFF -DLITE_WITH_X86=ON ${common_flags} -DLITE_WITH_PROFILE=ON

    # Compile and execute the gen_code related test, so it will generate some code, and make the compilation reasonable.
    make test_gen_code_lite -j$NUM_CORES_FOR_COMPILE
    make test_cxx_api_lite -j$NUM_CORES_FOR_COMPILE
    ctest -R test_cxx_api_lite
    ctest -R test_gen_code_lite
    make test_generated_code -j$NUM_CORES_FOR_COMPILE
}

function cmake_gpu {
    prepare_for_codegen
    cmake .. " -DWITH_GPU=ON {common_flags} -DLITE_WITH_GPU=ON"
}

function check_style {
    export PATH=/usr/bin:$PATH
    #pre-commit install
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

function build_single {
    #make $1 -j$(expr $(nproc) - 2)
    make $1 -j$NUM_CORES_FOR_COMPILE
}

function build {
    make lite_compile_deps -j $NUM_CORES_FOR_COMPILE
}

# It will eagerly test all lite related unittests.
function test_lite {
    local file=$1
    echo "file: ${file}"

    for _test in $(cat $file); do
        ctest -R $_test -V
    done
}

# Build the code and run lite server tests. This is executed in the CI system.
function build_test_server {
    mkdir -p ./build
    cd ./build
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/paddle/build/third_party/install/mklml/lib"
    cmake_x86_for_CI
    build
    test_lite $TESTS_FILE
}

# test_arm_android <some_test_name> <adb_port_number>
function test_arm_android {
    test_name=$1
    port=$2
    if [[ "${test_name}x" == "x" ]]; then
        echo "test_name can not be empty"
        exit 1
    fi
    if [[ "${port}x" == "x" ]]; then
        echo "Port can not be empty"
        exit 1
    fi

    echo "test name: ${test_name}"
    adb_work_dir="/data/local/tmp"
    skip_list="test_model_parser_lite" # add more with space
    [[ $skip_list =~ (^|[[:space:]])$test_name($|[[:space:]]) ]] && continue || echo 'skip $test_name'
    testpath=$(find ./paddle/fluid -name ${test_name})
    adb -s emulator-${port} push ${testpath} ${adb_work_dir}
    adb -s emulator-${port} shell chmod +x "${adb_work_dir}/${test_name}"
    adb -s emulator-${port} shell "./${adb_work_dir}/${test_name}"
}

# Build the code and run lite arm tests. This is executed in the CI system.
function build_test_arm {
    port_armv8=5554
    port_armv7=5556

    adb kill-server
    adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
    # start android arm64-v8a armeabi-v7a emulators first
    echo n | avdmanager create avd -f -n paddle-armv8 -k "system-images;android-24;google_apis;arm64-v8a"
    echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv8 -noaudio -no-window -gpu off -verbose -port ${port_armv8} &
    sleep 1m
    echo n | avdmanager create avd -f -n paddle-armv7 -k "system-images;android-24;google_apis;armeabi-v7a"
    echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv7 -noaudio -no-window -gpu off -verbose -port ${port_armv7} &
    sleep 1m

    cur_dir=$(pwd)

    for os in "android" "armlinux" ; do
        for abi in "arm64-v8a" "armeabi-v7a" "armeabi-v7a-hf" ; do
            # TODO(TJ): enable compile on v7-hf on andorid and all v7 on armlinux
            if [[ ${abi} == "armeabi-v7a-hf" ]]; then
                echo "armeabi-v7a-hf is not supported on both android and armlinux"
                continue
            fi

            if [[ ${os} == "armlinux" && ${abi} == "armeabi-v7a" ]]; then
                echo "armeabi-v7a is not supported on armlinux yet"
                continue
            fi

            build_dir=$cur_dir/build.lite.${os}.${abi}
            mkdir -p $build_dir
            cd $build_dir

            cmake_arm ${os} ${abi}
            build $TESTS_FILE

            # armlinux need in another docker
            # TODO(TJ): enable test with armlinux
            if [[ ${os} == "android" ]]; then
                adb_abi=${abi}
                if [[ ${adb_abi} == "armeabi-v7a-hf" ]]; then
                    adb_abi="armeabi-v7a"
                fi
                if [[ ${adb_abi} == "armeabi-v7a" ]]; then
                    # skip all armv7 tests
                    # TODO(TJ): enable test with armv7
                    continue
                fi
                local port=
                if [[ ${adb_abi} == "armeabi-v7a" ]]; then
                    port=${port_armv7}
                fi

                if [[ ${adb_abi} == "arm64-v8a" ]]; then
                    port=${port_armv8}
                fi
                echo "test file: ${TESTS_FILE}"
                for _test in $(cat $TESTS_FILE); do
                    test_arm_android $_test $port
                done
            fi
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
    echo -e "--arm_os=<os> --arm_abi=<abi> cmake_arm: run cmake with ARM mode"
    echo
    echo -e "build: compile the tests"
    echo -e "--test_name=<test_name> build_single: compile single test"
    echo
    echo -e "test_server: run server tests"
    echo -e "--test_name=<test_name> --adb_port_number=<adb_port_number> test_arm_android: run arm test"
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
            --test_name=*)
                TEST_NAME="${i#*=}"
                shift
                ;;
            --arm_os=*)
                ARM_OS="${i#*=}"
                shift
                ;;
            --arm_abi=*)
                ARM_ABI="${i#*=}"
                shift
                ;;
            --arm_port=*)
                ARM_PORT="${i#*=}"
                shift
                ;;
            build)
                build $TESTS_FILE
                build $LIBS_FILE
                shift
                ;;
            build_single)
                build_single $TEST_NAME
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
                cmake_arm $ARM_OS $ARM_ABI
                shift
                ;;
            test_server)
                test_lite $TESTS_FILE
                shift
                ;;
            test_arm_android)
                test_arm_android $TEST_NAME $ARM_PORT
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
            check_need_ci)
                check_need_ci
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

main $@
 
