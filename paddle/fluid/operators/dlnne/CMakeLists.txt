# compile flags
<<<<<<< HEAD
set(DLNNE_FLAGS -Wno-error=non-virtual-dtor -Wno-error=unused-variable
                -Wno-error=attributes ${fsanitize})
=======
set(DLNNE_FLAGS -Wno-error=non-virtual-dtor -Wno-error=attributes ${fsanitize})
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
foreach(flag ${DLNNE_FLAGS})
  safe_set_cflag(CMAKE_C_FLAGS ${flag})
  safe_set_cxxflag(CMAKE_CXX_FLAGS ${flag})
endforeach()

# add nne
find_path(
  DLNNE_INCLUDE_DIR dlnne.h
<<<<<<< HEAD
  PATHS $ENV{SOFTWARE_SOURCE_DIR} $ENV{SOFTWARE_SOURCE_DIR}/driver/nne/include
=======
  PATHS $ENV{DL_SDK_DIR} $ENV{DL_SDK_DIR}/include/dlnne
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  NO_DEFAULT_PATH)

find_library(
  DLNNE_LIB libdlnne.so
<<<<<<< HEAD
  PATHS $ENV{SOFTWARE_BUILD_DIR} $ENV{SOFTWARE_BUILD_DIR}/driver/nne
  NO_DEFAULT_PATH)

find_path(CUDA_INCLUDE_DIR cuda.h
          $ENV{SOFTWARE_BUILD_DIR}/llvm-project-10/cuda/include)

find_library(
  CURT_LIB libcurt.so
  PATHS $ENV{SOFTWARE_BUILD_DIR}
        $ENV{SOFTWARE_BUILD_DIR}/llvm-project-10/cuda/lib
=======
  PATHS $ENV{DL_SDK_DIR} $ENV{DL_SDK_DIR}/lib
  NO_DEFAULT_PATH)

find_path(CUDA_INCLUDE_DIR cuda.h $ENV{DL_SDK_DIR}/include)

find_library(
  CURT_LIB libcurt.so
  PATHS $ENV{DL_SDK_DIR} $ENV{DL_SDK_DIR}/lib
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
