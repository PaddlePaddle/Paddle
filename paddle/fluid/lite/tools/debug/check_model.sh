#!/bin/bash

############################# Arguments ############################
# For both cpp & python
BUILD_ROOT_DIR=""                 # Cmake build root path, for LD_LIBRARY_PATH
MODEL_DIR=""                      # Model dir path
INPUT_FILE=""                     # Input data file, only the first record will be used. 
                                  # If the path is empty, then all-ones input will be used.
CPP_TOPO_FILE=./topo_file.txt     # Runtime program topology info. Write by Cpp-debug-tool and Read by Py-debug-tool
CPP_TENSOR_FILE=./tensor_cpp.txt  # Store Cpp-debug-tool's tensor outputs int runtime topology order.
                                  # Write by Cpp-debug-tool and Read by Py-debug-tool 
TENSOR_NAMES=""                   # If is not empty, then only dump the tensor fo arguments whoes name is 
                                  # in tensor names. Separate by ','.
TENSOR_OUTPUT_LENGTH=-1           # Output tensor data length. Tensor's dim size will be used if this value < 0.

# For Cpp debug tools
CPP_OUTPUT_TOPO=1                 # If output topology info or not.
CPP_OUTPUT_VARS=1                 # If output TmpVar' tensor or not.
CPP_OUTPUT_WEIGHTS=1              # If output WeightVar' tensor or not.
CPP_ARM_THREAD_NUM=1              # ARM thread num. Used by ARM device info. 
                                  # Only be used by compile option - LITE_WITH_ARM

# For python debug tools
PY_THRESHOLD=0.00001              # The numerical lower bound  be used to judge [Cpp vs Py] runtime model diff.
PY_TENSOR_FILE=./tensor_py.txt    # Store Py-debug-tool's tensor outputs.
PY_OUTPUT_FILE=./diff.txt         # Store model different op/var info for debug.
PY_ONLY_OUTPUT_FIRST_DIFF=1       # If only output the first different var's info in runtime topology order or not.
PY_OUTPUT_TENSOR=1                # If output var' tensor in CPP_TENSOR_FILE/TENSOR_NAMES or not.

############################# MAIN #################################
function print_usage {
    echo -e "\nUSAGE:"
    echo -e "debug_cpp_stage -> debug_py_stage"
    echo
    echo "----------------------------------------"
    echo -e "debug_cpp_stage:"
    echo -e "run_debug.sh [--option=value]* debug_cpp_stage"
    echo -e "See run_debug.sh#run_cpp_debug_tool for detail"
    echo
    echo -e "debug_py_stage:"
    echo -e "run_debug.sh [--option=value]* debug_py_stage"
    echo -e "See run_debug.sh#run_py_debug_tool for detail"
    echo "----------------------------------------"
}

function check_enviroment {
    if [ "X${BUILD_ROOT_DIR}" == "X" ]; then
	echo -e "\nOption: --build_root_dir=xxx is required.\n";
	exit 1
    fi 
    if [ "X${MODEL_DIR}" == "X" ]; then
	echo -e "\nOption: --model_dir=xxx is required.\n";
	exit 1
    fi 
}

function run_cpp_debug_tool {
    check_enviroment

    local tool_name="lite_model_debug_tool"
    local tool_path=$(find ${BUILD_ROOT_DIR} -type f -name ${tool_name})
    if [ "X${tool_path}" == "X" ]; then
	echo -e "\nERROR: ${tool_name} not found in ${BUILD_ROOT_DIR}.\n"
	exit 1
    fi
    echo "Find Cpp-debug-tool path: ${tool_path}"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$BUILD_ROOT_DIR/third_party/install/mklml/lib"
    ${tool_path} \
        --model_dir=$MODEL_DIR                         \
        --input_file=$INPUT_FILE                       \
        --topo_output_file=$CPP_TOPO_FILE              \
        --output_topo=$CPP_OUTPUT_TOPO                 \
        --tensor_output_file=$CPP_TENSOR_FILE          \
        --output_vars=$CPP_OUTPUT_VARS                 \
        --output_weights=$CPP_OUTPUT_WEIGHTS           \
        --tensor_names=$TENSOR_NAMES                   \
        --tensor_output_length=$TENSOR_OUTPUT_LENGTH   \
        --arm_thread_num=$CPP_ARM_THREAD_NUM
}

function run_py_debug_tool {
    check_enviroment

    local tool_name="analysis_tool.py"
    local tool_path=$(find ${BUILD_ROOT_DIR} -type f -name ${tool_name})
    if [ "X${tool_path}" == "X" ]; then
	echo -e "\nERROR: ${tool_name} not found in ${BUILD_ROOT_DIR}.\n"
	return
    fi
    echo "Find Py-debug-tool path: ${tool_path}"
    python ${tool_path} \
        --model_dir=$MODEL_DIR                         \
        --input_file=$INPUT_FILE                       \
        --topo_file=$CPP_TOPO_FILE                     \
        --tensor_file=$CPP_TENSOR_FILE                 \
        --tensor_names=$TENSOR_NAMES                   \
        --output_tensor=$PY_OUTPUT_TENSOR              \
        --tensor_output_file=$PY_TENSOR_FILE           \
        --tensor_output_length=$TENSOR_OUTPUT_LENGTH   \
        --only_first=$PY_ONLY_OUTPUT_FIRST_DIFF        \
        --output_file=$PY_OUTPUT_FILE                  \
        --threshold=$PY_THRESHOLD
}

function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            --model_dir=*)
                MODEL_DIR="${i#*=}"
                shift
                ;;
            --input_file=*)
                INPUT_FILE="${i#*=}"
                shift
                ;;
            --cpp_topo_file=*)
                CPP_TOPO_FILE="${i#*=}"
                shift
                ;;
            --cpp_tensor_file=*)
                CPP_TENSOR_FILE="${i#*=}"
                shift
                ;;
            --tensor_names=*)
                TENSOR_NAMES="${i#*=}"
                shift
                ;;
            --tensor_output_length=*)
                TENSOR_OUTPUT_LENGTH="${i#*=}"
                shift
                ;;
            --cpp_output_vars=*)
                CPP_OUTPUT_VARS="${i#*=}"
                shift
                ;;
            --cpp_output_weights=*)
                CPP_OUTPUT_WEIGHTS="${i#*=}"
                shift
                ;;
            --py_threshold=*)
                PY_THRESHOLD="${i#*=}"
                shift
                ;;
            --py_tensor_file=*)
                PY_TENSOR_FILE="${i#*=}"
                shift
                ;;
            --py_output_file=*)
                PY_OUTPUT_FILE="${i#*=}"
                shift
                ;;
            --py_only_output_first_diff=*)
                PY_ONLY_OUTPUT_FIRST_DIFF="${i#*=}"
                shift
                ;;
            --py_output_tensor=*)
                PY_OUTPUT_TENSOR="${i#*=}"
                shift
                ;;
            --build_root_dir=*)
                BUILD_ROOT_DIR="${i#*=}"
                shift
                ;;
            debug_cpp_stage)
                run_cpp_debug_tool
                shift
                ;;
            debug_py_stage)
                run_py_debug_tool
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
