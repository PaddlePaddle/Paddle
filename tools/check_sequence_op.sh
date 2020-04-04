#!/bin/bash

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"

function check_sequnece_op_unitests(){
    check_white_list_file=$1
    function_grep=$2
    INVALID_SEQUENCE_OP_UNITTEST=""
    all_sequence_ops=`grep 'OP(sequence_' ${PADDLE_ROOT}/build/paddle/fluid/pybind/pybind.h | grep -Ev '^$' | cut -d'(' -f 2 | cut -d')' -f 1`
    for op_name in ${all_sequence_ops}; do
        in_white_list=`python ${PADDLE_ROOT}/${check_white_list_file} ${op_name}`
        if [ "${in_white_list}" == "True" ]; then
            continue
        fi
        unittest_file="python/paddle/fluid/tests/unittests/sequence/test_${op_name}.py"
        if [ ! -f "${PADDLE_ROOT}/${unittest_file}" ]; then
            INVALID_SEQUENCE_OP_UNITTEST="${INVALID_SEQUENCE_OP_UNITTEST}${unittest_file} (unittest file does not exists)\n"
            continue
        fi
        batch_size_1_funtion_calls=`grep ${function_grep} ${PADDLE_ROOT}/${unittest_file} || true`
        if [ "${batch_size_1_funtion_calls}" == "" ]; then
            INVALID_SEQUENCE_OP_UNITTEST="${INVALID_SEQUENCE_OP_UNITTEST}${unittest_file} (missing required function call)\n"
        fi
    done
    echo ${INVALID_SEQUENCE_OP_UNITTEST}
}

check_white_list_file="python/paddle/fluid/tests/unittests/white_list/check_op_sequence_batch_1_input_white_list.py"
function_grep="self.get_sequence_batch_size_1_input("
INVALID_SEQUENCE_OP_UNITTEST=$(check_sequnece_op_unitests ${check_white_list_file} ${function_grep})
if [ "${INVALID_SEQUENCE_OP_UNITTEST}" != "" ]; then
    echo "************************************"
    echo -e "It is required to include batch size 1 LoDTensor input in sequence OP test, please use self.get_sequence_batch_size_1_input() method."
    echo -e "For more information, please refer to [https://github.com/PaddlePaddle/Paddle/wiki/It-is-required-to-include-LoDTensor-input-with-batch_size=1-in-sequence-OP-test]."
    echo -e "Please check the following unittest files:\n${INVALID_SEQUENCE_OP_UNITTEST}"
    echo "************************************"
    exit 1
fi

check_white_list_file="python/paddle/fluid/tests/unittests/white_list/check_op_sequence_instance_0_input_white_list.py"
function_grep="self.get_sequence_instance_size_0_input("
INVALID_SEQUENCE_OP_UNITTEST=$(check_sequnece_op_unitests ${check_white_list_file} ${function_grep})
if [ "${INVALID_SEQUENCE_OP_UNITTEST}" != "" ]; then
    echo "************************************"
    echo -e "It is required to include instance size 0 LoDTensor input in sequence OP test, please use self.get_sequence_instance_size_0_input() method."
    echo -e "For more information, please refer to [https://github.com/PaddlePaddle/Paddle/wiki/It-is-required-to-include-LoDTensor-input-with-instance_size=0-in-sequence-OP-test]. "
    echo -e "Please check the following unittest files:\n${INVALID_SEQUENCE_OP_UNITTEST}"
    echo "************************************"
    exit 1
fi
