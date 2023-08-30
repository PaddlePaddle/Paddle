if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pessimizing-move")
endif()

add_subdirectory(details)

if(WITH_MKLDNN)
  set(mkldnn_quantizer_cfg mkldnn_quantizer_config)
  set(mkldnn_quantizer_src ${CMAKE_CURRENT_SOURCE_DIR}/mkldnn_quantizer.cc)
  cc_library(
    ${mkldnn_quantizer_cfg}
    SRCS mkldnn_quantizer_config.cc
    DEPS lod_tensor paddle_pass_builder)
  set(mkldnn_quantizer_cfg
      ${mkldnn_quantizer_cfg}
      PARENT_SCOPE)
endif()

cc_library(
  paddle_infer_contrib
  SRCS paddle_infer_contrib.cc
  DEPS zero_copy_tensor)
cc_library(
  paddle_pass_builder
  SRCS paddle_pass_builder.cc
  DEPS framework_proto)

set(paddle_inference_api_deps
    reset_tensor_array
    paddle_infer_contrib
    paddle_pass_builder
    zero_copy_tensor
    trainer_desc_proto
    custom_operator
    lod_tensor
    scope)

if(WITH_CRYPTO)
  list(APPEND paddle_inference_api_deps paddle_crypto)
endif()
if(WITH_CUSTOM_DEVICE)
  set(paddle_inference_api_deps ${paddle_inference_api_deps} phi)
endif()

if(WIN32)
  cc_library(
    paddle_inference_api
    SRCS api.cc api_impl.cc helper.cc
    DEPS executor ${paddle_inference_api_deps})
else()
  cc_library(
    paddle_inference_api
    SRCS api.cc api_impl.cc helper.cc
    DEPS executor paddle_inference_io ${paddle_inference_api_deps})
endif()

cc_library(
  analysis_config
  SRCS analysis_config.cc
  DEPS ${mkldnn_quantizer_cfg} paddle_inference_api paddle_pass_builder
       table_printer utf8proc)

if(WIN32)
  target_link_libraries(paddle_inference_api phi)
endif()

set(inference_deps ${analysis_deps} paddle_inference_api analysis
                   analysis_config naive_executor ${GLOB_PASS_LIB})

if(WITH_GPU AND TENSORRT_FOUND)
  set(inference_deps ${inference_deps} tensorrt_engine tensorrt_converter)
endif()

if(WITH_ONNXRUNTIME)
  cc_library(
    analysis_predictor
    SRCS analysis_predictor.cc onnxruntime_predictor.cc resource_manager.cc
         infer_context.cc ${mkldnn_quantizer_src}
    DEPS ${inference_deps}
         zero_copy_tensor
         ir_pass_manager
         op_compatible_info
         infer_io_utils
         model_utils
         onnxruntime
         paddle2onnx
         fleet_executor)
else()
  cc_library(
    analysis_predictor
    SRCS analysis_predictor.cc resource_manager.cc infer_context.cc
         ${mkldnn_quantizer_src}
    DEPS ${inference_deps}
         zero_copy_tensor
         ir_pass_manager
         op_compatible_info
         infer_io_utils
         model_utils
         fleet_executor)
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_paddle_inference_api)
endif()
