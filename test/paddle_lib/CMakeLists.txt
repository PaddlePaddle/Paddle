set(CMAKE_PREFIX_PATH "${PADDLE_BINARY_DIR}/paddle_install_dir")
find_package(Paddle REQUIRED)
include_directories(${PADDLE_INCLUDE_DIRS})

cc_test(
  test_paddle_lib
  SRCS test_paddle_lib.cc
  DEPS ${PADDLE_LIBRARIES})

if(WITH_GPU)
  cc_test(
    test_paddle_lib_gpu
    SRCS test_paddle_lib_gpu.cc
    DEPS ${PADDLE_LIBRARIES})
endif()
