set(cxx_api_lite_deps
  scope_lite optimizer_lite target_wrapper_host model_parser_lite program_lite)
if(LITE_WITH_CUDA)
    set(cxx_api_lite_deps ${cxx_api_lite_deps} kernels_cuda)
    cc_library(cxx_api_lite_cuda SRCS cxx_api.cc DEPS ${cxx_api_lite_deps} target_wrapper_cuda)
    nv_test(test_cxx_api_lite_cuda SRCS cxx_api_test.cc DEPS cxx_api_lite_cuda)
endif()

lite_cc_library(lite_api_test_helper SRCS lite_api_test_helper.cc
  DEPS scope_lite optimizer_lite target_wrapper_host model_parser_lite program_lite
       ${ops_lite} ${host_kernels}
  CUDA_DEPS kernels_cuda
  X86_DEPS ${x86_kernels}
  )
lite_cc_library(cxx_api_lite SRCS cxx_api.cc DEPS lite_api_test_helper)

set(light_api_deps
    scope_lite target_wrapper_host model_parser_lite program_lite)

if(LITE_WITH_CUDA)
    set(light_api_deps ${light_api_deps} target_wrapper_cuda)
endif()

lite_cc_library(light_api_lite SRCS light_api.cc
  DEPS ${light_api_deps} ${ops_lite} ${host_kernels}
  )

message(STATUS "get ops ${ops_lite}")
message(STATUS "get Host kernels ${host_kernels}")
message(STATUS "get ARM kernels ${arm_kernels}")

include(ExternalProject)
set(LITE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
        "A path setting inference demo download directories.")

if((NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK) AND WITH_TESTING)
    lite_cc_test(test_cxx_api_lite SRCS cxx_api_test.cc
       DEPS cxx_api_lite mir_passes
       ${ops_lite} ${host_kernels} ${x86_kernels}
       ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
            --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
    add_dependencies(test_cxx_api_lite extern_lite_download_lite_naive_model_tar_gz)
endif()


# These tests needs CLI arguments, and is not supported in ARM CI.
# TODO(Superjomn) support latter.
if(NOT LITE_ON_MOBILE)
    lite_cc_test(test_light_api SRCS light_api_test.cc
      DEPS light_api_lite mir_passes
      X86_DEPS ${x86_kernels}
      ARGS --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt
      SERIAL)

    lite_cc_test(test_apis_lite SRCS apis_test.cc
      DEPS cxx_api_lite light_api_lite ${ops_lite} mir_passes
      X86_DEPS ${x86_kernels}
      ARGS --model_dir=${LITE_MODEL_DIR}/lite_naive_model
          --optimized_model=${LITE_MODEL_DIR}/lite_naive_model_opt SERIAL)
endif()

lite_cc_binary(cxx_api_lite_bin SRCS cxx_api_bin.cc
    DEPS
    cxx_api_lite
    model_parser_lite
    target_wrapper_host
    mir_passes
    ${ops_lite} ${host_kernels}
    ARM_DEPS ${arm_kernels})
