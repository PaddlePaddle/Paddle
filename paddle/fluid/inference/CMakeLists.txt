set(FLUID_CORE_MODULES proto_desc paddle_memory lod_tensor executor prune init)

cc_library(paddle_fluid_api
    SRCS io.cc
    DEPS ${FLUID_CORE_MODULES} ${GLOB_OP_LIB})

# Create static library
get_property(fluid_modules GLOBAL PROPERTY FLUID_MODULES)
cc_library(paddle_fluid DEPS ${fluid_modules})

# Create shared library
cc_library(paddle_fluid_shared SHARED
    SRCS io.cc
    DEPS ARCHIVE_START ${GLOB_OP_LIB} ${FLUID_CORE_MODULES} ARCHIVE_END)
set_target_properties(paddle_fluid_shared PROPERTIES OUTPUT_NAME paddle_fluid)

if(WITH_TESTING)
  add_subdirectory(tests/book)
endif()
