set(FLUID_CORE_MODULES proto_desc paddle_memory lod_tensor executor prune init)

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
  ARCHIVE_END
  ${FLUID_CORE_MODULES})

SET_TARGET_PROPERTIES(paddle_fluid_shared PROPERTIES OUTPUT_NAME paddle_fluid)

# install library & headers
if(NOT WITH_C_API AND WITH_FLUID)
  install(FILES io.h DESTINATION include/paddle/inference)
  install(TARGETS paddle_fluid_shared DESTINATION lib)
endif()

if(WITH_TESTING)
  add_subdirectory(tests/book)
endif()
