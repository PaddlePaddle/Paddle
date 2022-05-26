unset GREP_OPTIONS
PADDLE_ROOT="/zhangbo/Paddle"

function get_precise_tests_map_file {
    cd ${PADDLE_ROOT}/build
    ln -sf $(which python3.8) /usr/local/bin/python
    ln -sf $(which pip3.8) /usr/local/bin/pip
    PATH=/usr/local/bin:${PATH}

    ### 初次运行需要执行如下：
    # pip uninstall -y paddlepaddle-gpu
    # pip install -r ${PADDLE_ROOT}/python/requirements.txt
    # pip install -r ${PADDLE_ROOT}/python/unittest_py/requirements.txt
    # pip install hypothesis
    # pip install -I ${PADDLE_ROOT}/build/python/dist/*whl

    cp ${PADDLE_ROOT}/build/python/paddle/fluid/tests/unittests/op_test.py ${PADDLE_ROOT}/build/python
    cp ${PADDLE_ROOT}/build/python/paddle/fluid/tests/unittests/testsuite.py ${PADDLE_ROOT}/build/python
    cp -r ${PADDLE_ROOT}/build/python/paddle/fluid/tests/unittests/white_list ${PADDLE_ROOT}/build/python

    # paddle所有的单测：all_test_cases
    all_test_cases=$(ctest -N -V)
    
    # paddle所有显存为0且去除不能并发的单测：all_mem0_uts
    all_mem0_uts=$(python /zhangbo/ut_logic/final_ut_parallel_rule.py)

    get_quickly_disable_ut||disable_ut_quickly='disable_ut'    # indicate whether the case was in quickly disable list

    # 显存为0的单测分类
    parallel_mem_0_uts='^job$'      # 显存为0，可并发执行的单测
    exclusive_mem_0_uts='^job$'     # 显存为0，exclusive类型单测
    multiple_card_mem_0_uts='^job$' # 显存为0，多卡单测

    # exclusive 及 multiple_card
    all_exclusive_uts='^job$'        # cases list which would be run exclusively
    all_multiple_card_uts='^job$'    # cases list which would take multiple GPUs, most cases would be two GPUs

set +x
    while read -r line; do
        if [[ "$line" == "" ]]; then
            continue
        fi
        
        read matchstr <<< $(echo "$line"|grep -oEi 'Test[ \t]+#')
        if [[ "$matchstr" == "" ]]; then
            # Any test case with LABELS property would be parse here
            # RUN_TYPE=EXCLUSIVE mean the case would run exclusively
            # RUN_TYPE=DIST mean the case would take two graph GPUs during runtime
            read is_exclusive <<< $(echo "$line"|grep -oEi "RUN_TYPE=EXCLUSIVE")
            read is_multicard <<< $(echo "$line"|grep -oEi "RUN_TYPE=DIST")
            read is_nightly <<< $(echo "$line"|grep -oEi "RUN_TYPE=NIGHTLY|RUN_TYPE=DIST:NIGHTLY|RUN_TYPE=EXCLUSIVE:NIGHTLY")
            continue
        fi

        read testcase <<< $(echo "$line"|grep -oEi "\w+$")

        if [[ "$is_nightly" != "" ]] && [ ${NIGHTLY_MODE:-OFF} == "OFF" ]; then
            echo $testcase" will only run at night."
            continue
        fi
        if [[ "$is_multicard" == "" ]]; then
            # trick: treat all test case with prefix "test_dist" as dist case, and would run on 2 GPUs
            read is_multicard <<< $(echo "$testcase"|grep -oEi "test_dist_")
        fi

        if [[ "$is_exclusive" != "" ]]; then
            if [[ $(echo $all_mem0_uts | grep -o "\^$testcase\\$") != "" ]]; then
                exclusive_mem_0_uts="$exclusive_mem_0_uts|^$testcase$"
            fi
            all_exclusive_uts="$all_exclusive_uts|^$testcase$"
        elif [[ "$is_multicard" != "" ]]; then
            if [[ $(echo $all_mem0_uts | grep -o "\^$testcase\\$") != "" ]]; then
                multiple_card_mem_0_uts="$multiple_card_mem_0_uts|^$testcase$"
            fi
            all_multiple_card_uts="$all_multiple_card_uts|^$testcase$"
        else
            if [[ $(echo $all_mem0_uts | grep -o "\^$testcase\\$") != "" ]]; then
                parallel_mem_0_uts="$parallel_mem_0_uts|^$testcase$"
            fi
        fi
        is_exclusive=''
        is_multicard=''
        is_nightly=''
        matchstr=''
        testcase=''
    done <<< "$all_test_cases";
set -x

    echo "parallel_mem_0_uts:"
    echo $parallel_mem_0_uts 
    echo $parallel_mem_0_uts > /zhangbo/ut_logic/parallel_mem_0_uts.txt

    echo 'exclusive_mem_0_uts:' 
    echo $exclusive_mem_0_uts 
    echo $exclusive_mem_0_uts > /zhangbo/ut_logic/exclusive_mem_0_uts.txt

    echo 'multiple_card_mem_0_uts:' 
    echo $multiple_card_mem_0_uts 
    echo $multiple_card_mem_0_uts > /zhangbo/ut_logic/multiple_card_mem_0_uts.txt

    echo 'all_exclusive_uts:' 
    echo $all_exclusive_uts 
    echo $all_exclusive_uts > /zhangbo/ut_logic/all_exclusive_uts.txt

    echo 'all_multiple_card_uts:' 
    echo $all_multiple_card_uts 
    echo $all_multiple_card_uts > /zhangbo/ut_logic/all_multiple_card_uts.txt

    # single_ut_startTime_s=`date +%s`
    # card_test "$parallel_mem_0_uts" 1 16            # run cases 24 job each time with single GPU
    # single_ut_endTime_s=`date +%s`

    # echo "parallel_mem_0_uts: $[ $single_ut_endTime_s - $single_ut_startTime_s ]s" 
    
    
}

# getting qucik disable ut list 
function get_quickly_disable_ut() {
    python -m pip install requests
    if disable_ut_quickly=$(python ${PADDLE_ROOT}/tools/get_quick_disable_lt.py); then
        echo "========================================="
        echo "The following unittests have been disabled:"
        echo ${disable_ut_quickly}
        echo "========================================="
    else
        disable_ut_quickly='disable_ut'
    fi
}

EXIT_CODE=0;
function caught_error() {
 for job in `jobs -p`; do
        # echo "PID => ${job}"
        if ! wait ${job} ; then
            echo "At least one test failed with exit code => $?" ;
            EXIT_CODE=1;
        fi
    done
}


tmp_dir=`mktemp -d`
echo 'tmp_dir: ' $tmp_dir
function card_test() {
    set -m
    ut_startTime_s=`date +%s` 

    bash xxx.sh &

    testcases=$1
    cardnumber=$2
    parallel_level_base=${CTEST_PARALLEL_LEVEL:-1}

    # get the CUDA device count, XPU device count is one
    if [ "${WITH_XPU}" == "ON" ];then
        CUDA_DEVICE_COUNT=1
    elif [ "${WITH_ASCEND_CL}" == "ON" ];then
        CUDA_DEVICE_COUNT=1
    elif [ "${WITH_ROCM}" == "ON" ];then
        CUDA_DEVICE_COUNT=$(rocm-smi -i | grep GPU | wc -l)
    elif [ "${WITH_MLU}" == "ON" ];then
        CUDA_DEVICE_COUNT=1
    elif [ "${WITH_IPU}" == "ON" ];then
        CUDA_DEVICE_COUNT=1
    else
        CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)  #2
    fi

    if (( $cardnumber == -1 ));then
        cardnumber=$CUDA_DEVICE_COUNT
    fi

    if (( $# > 2 )); then
        parallel_job=`expr $3 \* $parallel_level_base`
    else
        parallel_job=$parallel_level_base
    fi

    if [[ "$testcases" == "" ]]; then
        return 0
    fi

    trap 'caught_error' CHLD
    tmpfile_rand=`date +%s%N`
    NUM_PROC=$[CUDA_DEVICE_COUNT/$cardnumber]
    echo "****************************************************************"
    echo "***These unittests run $parallel_job job each time with $cardnumber GPU***"
    echo "****************************************************************"
    for (( i = 0; i < $NUM_PROC; i++ )); do
        # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
        # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
        cuda_list=()
        for (( j = 0; j < cardnumber; j++ )); do
            if [ $j -eq 0 ]; then
                    cuda_list=("$[i*cardnumber]")
                else
                    cuda_list="$cuda_list,$[i*cardnumber+j]"
            fi
        done
        tmpfile=$tmp_dir/$tmpfile_rand"_"$i
        if [ ${TESTING_DEBUG_MODE:-OFF} == "ON" ] ; then
            if [[ $cardnumber == $CUDA_DEVICE_COUNT ]]; then
        
                (ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" -V --timeout 120 -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            else 
                xxx startTime

                (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" --timeout 120 -V -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
                endTime
            fi
        else
            if [[ $cardnumber == $CUDA_DEVICE_COUNT ]]; then
               
                (ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            else
                (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            fi
        fi
    done
    wait; # wait for all subshells to finish
    ut_endTime_s=`date +%s`
    if (( $2 == -1 )); then
        echo "exclusive TestCases Total Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
    else
        echo "$2 card TestCases Total Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
    fi
    echo "$2 card TestCases finished!!!! "
    set +m
}

get_precise_tests_map_file