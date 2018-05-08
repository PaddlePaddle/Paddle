set(ENGINE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/engine.cc)
nv_library(tensorrt_engine SRCS engine.h engine.cc helper.h io_converter.cc ../engine.h)
nv_test(test_tensorrt SRCS test_tensorrt.cc DEPS dynload_cuda device_context dynamic_loader)
nv_test(test_tensorrt_engine SRCS test_engine.cc DEPS dynload_cuda tensorrt_engine)
nv_test(test_io_converter SRCS test_io_converter.cc DEPS dynload_cuda dynamic_loader lod_tensor tensorrt_engine)

add_subdirectory(convert)
