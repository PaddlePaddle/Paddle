# compile flags
set(DLNNE_FLAGS -Wno-error=non-virtual-dtor -Wno-error=unused-variable
                -Wno-error=attributes ${fsanitize})
foreach(flag ${DLNNE_FLAGS})
  safe_set_cflag(CMAKE_C_FLAGS ${flag})
  safe_set_cxxflag(CMAKE_CXX_FLAGS ${flag})
endforeach()

# add nne
find_path(
  DLNNE_INCLUDE_DIR dlnne.h
  PATHS $ENV{DL_SDK_DIR} $ENV{DL_SDK_DIR}/include/dlnne
  NO_DEFAULT_PATH)

find_library(
  DLNNE_LIB libdlnne.so
  PATHS $ENV{DL_SDK_DIR} $ENV{DL_SDK_DIR}/lib
  NO_DEFAULT_PATH)

find_path(CUDA_INCLUDE_DIR cuda.h $ENV{DL_SDK_DIR}/include)

find_library(
  CURT_LIB libcurt.so
  PATHS $ENV{DL_SDK_DIR} $ENV{DL_SDK_DIR}/lib
  NO_DEFAULT_PATH)

message("DLNNE_INCLUDE_DIR: "${DLNNE_INCLUDE_DIR})
message("DLNNE_LIB: "${DLNNE_LIB})
message("CUDA_INCLUDE_DIR: "${CUDA_INCLUDE_DIR})
message("CURT_LIB: "${CURT_LIB})

include_directories("${DLNNE_INCLUDE_DIR}")
include_directories("${CUDA_INCLUDE_DIR}")

op_library(
  dlnne_engine_op
  DEPS
  ${GLOB_OPERATOR_DEPS}
  framework_proto
  device_context
  op_registry
  scope)

#message("PYBIND_FILE:${pybind_file}")
#file(APPEND ${pybind_file} "USE_NO_KERNEL_OP(dlnne_engine);\n")
#endif()

target_link_libraries(dlnne_engine_op ${DLNNE_LIB} ${CURT_LIB})

cc_test(
  test_dlnne_engine_op
  SRCS dlnne_engine_op_test.cc
  DEPS dlnne_engine_op analysis)
