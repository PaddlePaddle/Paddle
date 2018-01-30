set(FLUID_CORE_MODULES proto_desc paddle_memory executor prune init)

cc_library(paddle_fluid_api
    SRCS io.cc
    DEPS ${FLUID_CORE_MODULES} ${GLOB_OP_LIB})

# Merge all modules into a single static library
cc_library(paddle_fluid DEPS paddle_fluid_api ${FLUID_CORE_MODULES} ${GLOB_OP_LIB})

# Create shared library
add_library(paddle_fluid_shared SHARED io.cc)

target_circle_link_libraries(paddle_fluid_shared
  ARCHIVE_START
  ${GLOB_OP_LIB}
  ${FLUID_CORE_MODULES}
  ARCHIVE_END)

SET_TARGET_PROPERTIES(paddle_fluid_shared PROPERTIES OUTPUT_NAME paddle_fluid)

# install library & headers
set(lib_dir "${CMAKE_INSTALL_PREFIX}/paddle/inference")
add_custom_target(inference_lib DEPENDS paddle_fluid_shared
    COMMAND mkdir -p "${lib_dir}"
    COMMAND cp "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${lib_dir}"
    COMMAND cp "${CMAKE_CURRENT_BINARY_DIR}/libpaddle_fluid.so" "${lib_dir}"
)
add_custom_target(inference_lib_dist DEPENDS 
  inference_lib framework_lib memory_lib platform_lib string_lib
  gflags_lib glog_lib protobuf_lib eigen3_lib)

add_executable(example example.cc)
if(APPLE)
  set(OPTIONAL_LINK_FLAGS)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(OPTIONAL_LINK_FLAGS "-undefined dynamic_lookup")
  endif()
  target_link_libraries(example
      -Wl,-force_load paddle_fluid
      ${OPTIONAL_LINK_FLAGS}
      ${PTOOLS_LIB})
else()
  target_link_libraries(example
      -Wl,--start-group -Wl,--whole-archive paddle_fluid
      -Wl,--no-whole-archive -Wl,--end-group
      ${PTOOLS_LIB})
endif()
