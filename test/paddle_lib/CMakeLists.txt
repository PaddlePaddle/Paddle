set(CMAKE_PREFIX_PATH "${PADDLE_BINARY_DIR}/paddle_install_dir")
find_package(Paddle REQUIRED)
include_directories(${PADDLE_INCLUDE_DIRS})

cc_test(test_paddle_lib SRCS test_paddle_lib.cc)
target_link_libraries(test_paddle_lib ${PADDLE_LIBRARIES})

if(WITH_GPU)
  cc_test(test_paddle_lib_gpu SRCS test_paddle_lib_gpu.cc)
  target_link_libraries(test_paddle_lib_gpu ${PADDLE_LIBRARIES})
endif()
