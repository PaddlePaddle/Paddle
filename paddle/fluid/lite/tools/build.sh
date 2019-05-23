#!/bin/bash
set -ex

TESTS_FILE=""

readonly common_flags="-DWITH_LITE=ON -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF -DWITH_PYTHON=OFF -DWITH_TESTING=ON"
function cmake_cpu {
    cmake ..  -DWITH_GPU=OFF ${common_flags}
    make test_cxx_api_lite -j8
}

function cmake_gpu {
    cmake .. " -DWITH_GPU=ON {common_flags} -DLITE_WITH_GPU=ON"
    make test_cxx_api_lite -j8
}

function build {
    file=$1
    for _test in $(cat $file); do
        make $_test -j8
    done
}

# It will eagerly test all lite related unittests.
function test_lite {
    file=$1
    echo "file: ${file}"
    for _test in $(cat $file); do
        ctest -R $_test -V
    done
}


# Parse command line.
for i in "$@"; do
    case $i in
        -e=*|--tests=*)
            TESTS_FILE="${i#*=}"
            shift
            ;;
        -b|--build)
            build $TESTS_FILE
            shift
            ;;
        -c|--cmake)
            cmake_cpu
            shift
            ;;
        -t|--test)
            test_lite $TESTS_FILE
            shift
            ;;
        *)
            # unknown option
            ;;
        esac
done

echo "tests: ${TESTS_FILE}"


#build $TESTS_FILE
#test_lite $TESTS_FILE
