add_definitions(-DUSE_ARM_PLACE)
set(CMAKE_CXX_FLAGS "-std=c++11 -pie -fPIE -Wno-attributes ${CMAKE_CXX_FLAGS}")
if (NOT (${CMAKE_CXX_COMPILER} MATCHES "clang\\+\\+$"))
  set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
endif()
if (ANDROID)
  set(CMAKE_CXX_FLAGS "-llog ${CMAKE_CXX_FLAGS}")
endif()
if (IOS)
  set(CMAKE_CXX_FLAGS "-fembed-bitcode ${CMAKE_CXX_FLAGS}")
endif()

set(PADDLE_LITE_LIB paddle-lite)
set(PADDLE_LITE_SRCS api.cc api_anakin_engine.cc)

set(PADDLE_LITE_PATH ${PADDLE_SOURCE_DIR}/paddle/fluid/inference/lite)
include_directories(${PADDLE_LITE_PATH} ${PADDLE_LITE_PATH}/output
                    ${PADDLE_LITE_PATH}/output/saber)

if (BUILD_SHARED_LIBS)
  add_library(${PADDLE_LITE_LIB} SHARED ${PADDLE_LITE_SRCS})
else()
  add_library(${PADDLE_LITE_LIB} STATIC ${PADDLE_LITE_SRCS})
endif(BUILD_SHARED_LIBS)
#target_link_libraries(${PADDLE_LITE_LIB} )

#add_library(anakin SHARED IMPORTED)
#set_target_properties(anakin PROPERTIES IMPORTED_LOCATION
#                      ${PADDLE_LITE_PATH}/output/libanakin.so)
add_library(anakin STATIC IMPORTED)
set_target_properties(anakin PROPERTIES IMPORTED_LOCATION
                      ${PADDLE_LITE_PATH}/output/libanakin_static.a)
add_library(saber_common STATIC IMPORTED)
set_target_properties(saber_common PROPERTIES IMPORTED_LOCATION
                      ${PADDLE_LITE_PATH}/output/libanakin_saber_common.a)
add_library(protobuf STATIC IMPORTED)
set_target_properties(protobuf PROPERTIES IMPORTED_LOCATION
                      ${PADDLE_LITE_PATH}/output/protobuf/lib/libprotobuf.a)

add_executable(test-benchmark benchmark/benchmark.cc)
target_link_libraries(test-benchmark paddle-lite "-Wl,--whole-archive" saber_common anakin "-Wl,--no-whole-archive" protobuf)
