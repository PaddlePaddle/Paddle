lite_cc_library(debug_utils_lite SRCS debug_utils.cc DEPS op_params_lite)

lite_cc_binary(lite_model_debug_tool SRCS model_debug_tool.cc
    DEPS
    cxx_api_lite
    debug_utils_lite
    model_parser_lite
    target_wrapper_host
    mir_passes
    ${ops_lite} ${host_kernels}
    X86_DEPS ${x86_kernels}
    ARM_DEPS ${arm_kernels}
    CL_DEPS ${opencl_kernels})
