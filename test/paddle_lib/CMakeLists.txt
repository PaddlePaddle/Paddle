project(test_paddle_lib)

get_filename_component(PADDLE_BINARY_DIR "${CMAKE_CURRENT_LIST_FILE}/../.."
                       ABSOLUTE)
list(APPEND CMAKE_PREFIX_PATH "${PADDLE_BINARY_DIR}/paddle_install_dir")
message(CMAKE_PREFIX_PATH)
find_package(Paddle REQUIRED)
include_directories(${PADDLE_INCLUDE_DIRS})

add_executable(test_paddle_lib test_paddle_lib.cc)
target_link_libraries(test_paddle_lib ${PADDLE_LIBRARIES})

if(WITH_GPU)
  add_executable(test_paddle_lib_gpu test_paddle_lib_gpu.cc)
  target_link_libraries(test_paddle_lib_gpu ${PADDLE_LIBRARIES})
endif()
