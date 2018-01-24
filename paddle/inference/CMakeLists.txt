set(FLUID_CORE_MODULES proto_desc paddle_memory executor prune init)

cc_library(paddle_fluid_api
    SRCS inference.cc
    DEPS ${FLUID_CORE_MODULES} ${GLOB_OP_LIB})

if(WITH_GPU)
  set(GPU_DEPS gpu_info dynload_cuda)
endif()
# Merge all modules into a single static library
cc_library(paddle_fluid DEPS paddle_fluid_api 
  framework_proto ddim tensor place scope threadpool attribute 
  shape_inference cpu_info op_registry operator op_proto_maker op_info 
  data_transform data_device_transform data_type_transform data_layout_transform
  selected_rows lod_tensor lod_rank_table dynload_warpctc dynamic_loader 
  device_context profiler
  ${GPU_DEPS}
  ${OP_MATH_MODULES}
  ${FLUID_CORE_MODULES} 
  ${GLOB_OP_LIB})

# Create shared library
add_library(paddle_fluid_shared SHARED inference.cc)

target_circle_link_libraries(paddle_fluid_shared
  ARCHIVE_START
  ${GLOB_OP_LIB}
  ARCHIVE_END
  ${FLUID_CORE_MODULES})

SET_TARGET_PROPERTIES(paddle_fluid_shared PROPERTIES OUTPUT_NAME paddle_fluid)

# install library & headers
if(NOT WITH_C_API AND WITH_FLUID)
  install(FILES inference.h DESTINATION include/paddle/inference)
  install(TARGETS paddle_fluid_shared paddle_fluid DESTINATION lib)
endif()

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
