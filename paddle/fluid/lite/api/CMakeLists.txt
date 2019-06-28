set(cxx_api_lite_deps
  scope_lite optimizer_lite target_wrapper_host model_parser_lite program_lite)
if(LITE_WITH_CUDA)
    set(cxx_api_lite_deps ${cxx_api_lite_deps} kernels_cuda)
    lite_cc_library(cxx_api_lite_cuda SRCS cxx_api.cc DEPS ${cxx_api_lite_deps} target_wrapper_cuda)
    nv_test(test_cxx_api_lite_cuda SRCS cxx_api_test.cc DEPS cxx_api_lite_cuda)
endif()

lite_cc_library(place_lite SRCS paddle_place.cc DEPS glog)

lite_cc_library(lite_api_test_helper SRCS lite_api_test_helper.cc
  DEPS scope_lite optimizer_lite target_wrapper_host model_parser_lite program_lite
       ${ops_lite} ${host_kernels}
  CUDA_DEPS kernels_cuda
  X86_DEPS ${x86_kernels})

set(light_api_deps
    scope_lite target_wrapper_host model_parser_lite program_lite)

if(LITE_WITH_CUDA)
    set(light_api_deps ${light_api_deps} target_wrapper_cuda)
endif()

message(STATUS "get ops ${ops_lite}")
message(STATUS "get Host kernels ${host_kernels}")
message(STATUS "get ARM kernels ${arm_kernels}")

lite_cc_library(cxx_api_lite
                SRCS cxx_api.cc
                DEPS ${cxx_api_lite_deps} ${ops_lite} ${host_kernels} program_lite
                X86_DEPS ${x86_kernels} operator
                ARM_DEPS ${arm_kernels}
                CL_DEPS ${opencl_kenrels})

lite_cc_library(light_api_lite SRCS light_api.cc
        DEPS scope_lite target_wrapper_host model_parser_lite
            ${light_api_deps} ${ops_lite} ${host_kernels} program_lite
        CUDA_DEPS target_wrapper_cuda
        X86_DEPS ${x86_kernels} operator
        ARM_DEPS ${arm_kernels}
        CL_DEPS ${opencl_kenrels})

include(ExternalProject)
set(LITE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
        "A path setting inference demo download directories.")

if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND WITH_TESTING)
    lite_cc_test(test_cxx_api_lite SRCS cxx_api_test.cc
       DEPS cxx_api_lite mir_passes lite_api_test_helper
       ${ops_lite} ${host_kernels} ${x86_kernels}
       ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
            --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
    add_dependencies(test_cxx_api_lite extern_lite_download_lite_naive_model_tar_gz)
    lite_cc_test(test_googlenet_lite SRCS test_googlenet_lite.cc
       DEPS cxx_api_lite mir_passes lite_api_test_helper
       ${ops_lite} ${host_kernels} ${x86_kernels}
       ARGS --model_dir=${LITE_MODEL_DIR}/googlenet)
    add_dependencies(test_googlenet_lite extern_lite_download_GoogleNet_inference_tar_gz)
endif()

if(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND WITH_TESTING)
    set(lite_model_test_DEPS cxx_api_lite mir_passes ${ops_lite} ${host_kernels} ${arm_kernels})

    lite_cc_test(test_mobilenetv1_lite SRCS mobilenetv1_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/paddle/fluid/lite/opencl
            --model_dir=${LITE_MODEL_DIR}/mobilenet_v1 SERIAL)
    add_dependencies(test_mobilenetv1_lite extern_lite_download_mobilenet_v1_tar_gz)

    lite_cc_test(test_mobilenetv2_lite SRCS mobilenetv2_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/paddle/fluid/lite/opencl
            --model_dir=${LITE_MODEL_DIR}/mobilenet_v2 SERIAL)
    add_dependencies(test_mobilenetv2_lite extern_lite_download_mobilenet_v2_relu_tar_gz)

    lite_cc_test(test_resnet50_lite SRCS resnet50_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/paddle/fluid/lite/opencl
            --model_dir=${LITE_MODEL_DIR}/resnet50 SERIAL)
    add_dependencies(test_resnet50_lite extern_lite_download_resnet50_tar_gz)

    lite_cc_test(test_inceptionv4_lite SRCS inceptionv4_test.cc
       DEPS ${lite_model_test_DEPS}
       CL_DEPS ${opencl_kernels}
       ARGS --cl_path=${CMAKE_SOURCE_DIR}/paddle/fluid/lite/opencl
            --model_dir=${LITE_MODEL_DIR}/inception_v4 SERIAL)
    add_dependencies(test_inceptionv4_lite extern_lite_download_inception_v4_simple_tar_gz)
endif()

# These tests needs CLI arguments, and is not supported in ARM CI.
# TODO(Superjomn) support latter.
lite_cc_test(test_light_api_lite SRCS light_api_test.cc
    DEPS light_api_lite program_lite mir_passes
    CL_DEPS ${opencl_kernels}
    ARGS --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

lite_cc_test(test_apis_lite SRCS apis_test.cc
  DEPS cxx_api_lite light_api_lite ${ops_lite}
  CL_DEPS ${opencl_kernels}
  X86_DEPS ${x86_kernels} operator
  ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
       --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)

lite_cc_library(paddle_api_lite SRCS paddle_api.cc DEPS op_params_lite)

#-----------------------------------------------------------------------------------------------------
# The final inference library for both CxxConfig and MobileConfig.
lite_cc_library(paddle_api_full SRCS cxx_api_impl.cc DEPS cxx_api_lite paddle_api_lite light_api_lite
  ${ops_lite}
  ARM_DEPS ${arm_kernels}
  CL_DEPS ${opencl_kernels})
# The final inference library for just MobileConfig.
lite_cc_library(paddle_api_light SRCS light_api_impl.cc DEPS light_api_lite paddle_api_lite mir_passes)

bundle_static_library(paddle_api_full paddle_api_full_bundled bundle_full_api)
bundle_static_library(paddle_api_light paddle_api_light_bundled bundle_light_api)
#-----------------------------------------------------------------------------------------------------


lite_cc_test(test_paddle_api_lite SRCS paddle_api_test.cc DEPS paddle_api_full paddle_api_light
  ${ops_lite}
  ARM_DEPS ${arm_kernels}
  CL_DEPS ${opencl_kernels}
  X86_DEPS ${x86_kernels}
  ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model SERIAL)

lite_cc_test(test_model_bin SRCS model_test.cc DEPS paddle_api_full paddle_api_light
  ${ops_lite}
  ARM_DEPS ${arm_kernels}
  CL_DEPS ${opencl_kernels}
  X86_DEPS ${x86_kernels})

if (WITH_TESTING)
    add_dependencies(test_paddle_api_lite extern_lite_download_lite_naive_model_tar_gz)
endif()

if (LITE_WITH_JAVA AND LITE_WITH_ARM)
    add_subdirectory(android)
endif()

#lite_cc_binary(cxx_api_lite_bin SRCS cxx_api_bin.cc
    #X86_DEPS operator
    #DEPS light_api_lite model_parser_lite target_wrapper_host mir_passes
    #ARM_DEPS ${arm_kernels})

lite_cc_binary(cxx_api_lite_bin SRCS cxx_api_bin_int8.cc
    DEPS
    cxx_api_lite
    model_parser_lite
    target_wrapper_host
    mir_passes
    ${ops_lite} ${host_kernels}
    ARM_DEPS ${arm_kernels}
    CL_DEPS ${opencl_kernels})
lite_cc_binary(model_optimize_tool SRCS model_optimize_tool.cc
    DEPS paddle_api_full
    CL_DEPS ${opencl_kernels})
