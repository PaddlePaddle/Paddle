add_subdirectory(memory)
add_subdirectory(platform)
add_subdirectory(framework)
add_subdirectory(imperative)
add_subdirectory(operators)
add_subdirectory(string)
add_subdirectory(pybind)

# NOTE: please add subdirectory inference at last.
add_subdirectory(inference)
add_subdirectory(train)

set(FLUID_FRAMEWORK_MODULES proto_desc memory lod_tensor executor data_feed_proto layer)
if(WIN32)
  sep_library(paddle_framework DEPS ${FLUID_FRAMEWORK_MODULES})
else(WIN32)
  cc_library(paddle_framework DEPS ${FLUID_FRAMEWORK_MODULES})
endif(WIN32)

cc_library(paddle_framework_shared
    SHARED SRCS ${CMAKE_CURRENT_SOURCE_DIR}/framework/executor.cc ${CMAKE_CURRENT_SOURCE_DIR}/framework/operator.cc ${CMAKE_CURRENT_SOURCE_DIR}/framework/c/c_api.cc ${CMAKE_CURRENT_SOURCE_DIR}/imperative/layer.cc
    DEPS ${FLUID_FRAMEWORK_MODULES})
get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
set_target_properties(paddle_framework_shared PROPERTIES OUTPUT_NAME paddle_framework)
target_link_libraries(paddle_framework_shared ${os_dependency_modules})

if (LINUX)
  set(FLUID_FRAMEWORK_SHARED_LIB
      ${PADDLE_BINARY_DIR}/paddle/fluid/libpaddle_framework.so
      CACHE INTERNAL "Fluid framework lib")
endif()

if (WIN32)
  set(FLUID_FRAMEWORK_SHARED_LIB
      ${PADDLE_BINARY_DIR}/paddle/fluid/libpaddle_framework.dll
      CACHE INTERNAL "Fluid framework lib")
endif()

if(APPLE)
  set(FLUID_FRAMEWORK_SHARED_LIB
      ${PADDLE_BINARY_DIR}/paddle/fluid/libpaddle_framework.dylib
      CACHE INTERNAL "Fluid framework lib")
endif()
