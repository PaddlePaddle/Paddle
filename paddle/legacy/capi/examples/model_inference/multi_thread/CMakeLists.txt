project(multi_thread)
cmake_minimum_required(VERSION 2.8)

find_package (Threads)

if(NOT PADDLE_ROOT)
  set(PADDLE_ROOT $ENV{PADDLE_ROOT} CACHE PATH "Paddle Path")
endif()
if(PADDLE_ROOT)
  include_directories(${PADDLE_ROOT}/include)
  link_directories(${PADDLE_ROOT}/lib)
endif()

set(CPU_SRCS main.c)
add_executable(${PROJECT_NAME} ${CPU_SRCS})
set_property(TARGET ${PROJECT_NAME} PROPERTY C_STANDARD 99)
target_link_libraries(${PROJECT_NAME}
                      -lpaddle_capi_shared
                      ${CMAKE_THREAD_LIBS_INIT})

find_package(CUDA QUIET)
if(CUDA_FOUND)
  set(GPU_SRCS main_gpu.c)
  cuda_add_executable(${PROJECT_NAME}_gpu ${GPU_SRCS})
  set_property(TARGET ${PROJECT_NAME}_gpu PROPERTY C_STANDARD 99)
  target_link_libraries(${PROJECT_NAME}_gpu
                        -lpaddle_capi_shared
                        ${CMAKE_THREAD_LIBS_INIT})
endif(CUDA_FOUND)
