if (NOT LITE_WITH_OPENCL)
    return()
endif()

lite_cc_library(cl_wrapper SRCS cl_wrapper.cxx)
lite_cc_library(cl_tool SRCS cl_tool.cc DEPS cl_wrapper)
lite_cc_library(cl_engine SRCS cl_engine.cc DEPS cl_tool)
lite_cc_library(cl_context SRCS cl_context.cc DEPS cl_engine)
lite_cc_library(cl_helper SRCS cl_helper.cc DEPS cl_context)
lite_cc_library(cl_image_converter SRCS cl_image_converter.cc DEPS lite_tensor)
lite_cc_library(cl_image SRCS cl_image.cc DEPS lite_tensor cl_image_converter cl_engine)
lite_cc_library(cl_caller SRCS cl_caller.cc  DEPS cl_helper cl_image)
lite_cc_test(test_cl_runtime SRCS cl_test.cc DEPS cl_helper cl_image cl_caller cl_wrapper
             ARGS --cl_path=${CMAKE_SOURCE_DIR}/paddle/fluid/lite/opencl)

add_dependencies(cl_wrapper opencl_clhpp)
