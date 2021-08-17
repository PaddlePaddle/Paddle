include(operators)

if (WITH_GPU OR WITH_ROCM)
    # resnet_unit needs cudnn 8.0 above
    if (NOT ${CUDNN_VERSION} VERSION_LESS 8000)
        op_library(resnet_unit_op)
        file(APPEND ${pybind_file} "USE_CUDA_ONLY_OP(resnet_unit);\n")
    endif()
endif()
