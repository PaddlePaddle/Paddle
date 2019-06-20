if (NOT LITE_WITH_CL)
    return()
endif()

find_library(opencl-lib
        NAMES OpenCL)
message(STATUS "The OpenCL library path : ${opencl-lib}")

add_compile_options(-fno-strict-aliasing)

cc_library(cl_tool SRCS cl_tool.cc)
cc_library(cl_half SRCS cl_half.cc)
cc_library(cl_engine SRCS cl_engine.cc DEPS cl_tool)
cc_library(cl_context SRCS cl_context.cc DEPS cl_engine)
cc_library(cl_helper SRCS cl_helper.cc DEPS cl_context)
cc_library(cl_image_converter SRCS cl_image_converter.cc DEPS cl_half lite_tensor)
cc_library(cl_image SRCS cl_image.cc DEPS cl_half lite_tensor cl_image_converter)
cc_test(test_cl_runtime SRCS cl_test.cc DEPS cl_engine cl_context)
target_link_libraries(test_cl_runtime ${opencl-lib})
