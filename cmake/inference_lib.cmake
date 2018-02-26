# make package for paddle fluid shared and static library
function(copy TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DSTS DEPS)
    cmake_parse_arguments(copy_lib "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    list(LENGTH copy_lib_SRCS copy_lib_SRCS_len)
    list(LENGTH copy_lib_DSTS copy_lib_DSTS_len)
    if(NOT ${copy_lib_SRCS_len} EQUAL ${copy_lib_DSTS_len})
        message(FATAL_ERROR "${TARGET} source numbers are not equal to destination numbers")
    endif()
    math(EXPR len "${copy_lib_SRCS_len} - 1")
    
    add_custom_target(${TARGET} DEPENDS ${copy_lib_DEPS})
    foreach(index RANGE ${len})
        list(GET copy_lib_SRCS ${index} src)
        list(GET copy_lib_DSTS ${index} dst)
        add_custom_command(TARGET ${TARGET} PRE_BUILD 
          COMMAND mkdir -p "${dst}"
          COMMAND cp -r "${src}" "${dst}"
          COMMENT "copying ${src} -> ${dst}")
    endforeach()
endfunction()

# third party
set(dst_dir "${CMAKE_INSTALL_PREFIX}/third_party/eigen3")
copy(eigen3_lib
  SRCS ${EIGEN_INCLUDE_DIR}/Eigen/Core ${EIGEN_INCLUDE_DIR}/Eigen/src ${EIGEN_INCLUDE_DIR}/unsupported/Eigen
  DSTS ${dst_dir}/Eigen ${dst_dir}/Eigen ${dst_dir}/unsupported
)

set(dst_dir "${CMAKE_INSTALL_PREFIX}/third_party/install/gflags")
copy(gflags_lib
  SRCS ${GFLAGS_INCLUDE_DIR} ${GFLAGS_LIBRARIES}
  DSTS ${dst_dir} ${dst_dir}/lib
)

set(dst_dir "${CMAKE_INSTALL_PREFIX}/third_party/install/glog")
copy(glog_lib
  SRCS ${GLOG_INCLUDE_DIR} ${GLOG_LIBRARIES}
  DSTS ${dst_dir} ${dst_dir}/lib
)

IF(NOT PROTOBUF_FOUND)
    set(dst_dir "${CMAKE_INSTALL_PREFIX}/third_party/install/protobuf")
    copy(protobuf_lib
      SRCS ${PROTOBUF_INCLUDE_DIR} ${PROTOBUF_LITE_LIBRARY}
      DSTS ${dst_dir} ${dst_dir}/lib
    )
ENDIF(NOT PROTOBUF_FOUND)

# paddle fluid module
set(src_dir "${PADDLE_SOURCE_DIR}/paddle/fluid")
set(dst_dir "${CMAKE_INSTALL_PREFIX}/paddle/fluid")
set(module "framework")
copy(framework_lib DEPS framework_py_proto 
  SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/details/*.h ${PADDLE_BINARY_DIR}/paddle/fluid/framework/framework.pb.h
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}/details ${dst_dir}/${module}
)

set(module "memory")
copy(memory_lib
  SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/detail/*.h
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}/detail
)

set(module "inference")
copy(inference_lib DEPENDS paddle_fluid_shared
  SRCS ${src_dir}/${module}/*.h ${PADDLE_BINARY_DIR}/paddle/fluid/inference/libpaddle_fluid.so
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}
)

set(module "platform")
copy(platform_lib
  SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/dynload/*.h ${src_dir}/${module}/details/*.h
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}/dynload ${dst_dir}/${module}/details
)

set(module "string")
copy(string_lib
  SRCS ${src_dir}/${module}/*.h ${src_dir}/${module}/tinyformat/*.h
  DSTS ${dst_dir}/${module} ${dst_dir}/${module}/tinyformat
)

add_custom_target(inference_lib_dist DEPENDS 
  inference_lib framework_lib memory_lib platform_lib string_lib
  gflags_lib glog_lib protobuf_lib eigen3_lib)
