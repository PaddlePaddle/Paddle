# make package for paddle fluid shared and static library
# third party
set(lib_dir "${CMAKE_INSTALL_PREFIX}/third_party/eigen3")
add_custom_target(eigen3_lib
    COMMAND mkdir -p "${lib_dir}/Eigen" "${lib_dir}/unsupported"
    COMMAND cp "${EIGEN_INCLUDE_DIR}/Eigen/Core" "${lib_dir}/Eigen"
    COMMAND cp -r "${EIGEN_INCLUDE_DIR}/Eigen/src" "${lib_dir}/Eigen"
    COMMAND cp -r "${EIGEN_INCLUDE_DIR}/unsupported/Eigen" "${lib_dir}/unsupported"
)

set(lib_dir "${CMAKE_INSTALL_PREFIX}/third_party/install/gflags")
add_custom_target(gflags_lib
    COMMAND mkdir -p "${lib_dir}/lib"
    COMMAND cp -r "${GFLAGS_INCLUDE_DIR}" "${lib_dir}"
    COMMAND cp "${GFLAGS_LIBRARIES}" "${lib_dir}/lib"
)

set(lib_dir "${CMAKE_INSTALL_PREFIX}/third_party/install/glog")
add_custom_target(glog_lib
    COMMAND mkdir -p "${lib_dir}/lib"
    COMMAND cp -r "${GLOG_INCLUDE_DIR}" "${lib_dir}"
    COMMAND cp "${GLOG_LIBRARIES}" "${lib_dir}/lib"
)

IF(NOT PROTOBUF_FOUND)
    set(lib_dir "${CMAKE_INSTALL_PREFIX}/third_party/install/protobuf")
    add_custom_target(protobuf_lib
        COMMAND mkdir -p "${lib_dir}/lib"
        COMMAND cp -r "${PROTOBUF_INCLUDE_DIR}" "${lib_dir}"
        COMMAND cp "${PROTOBUF_LITE_LIBRARY}" "${lib_dir}/lib"
    )
ENDIF(NOT PROTOBUF_FOUND)

# paddle fluid module
set(lib_dir "${CMAKE_INSTALL_PREFIX}/paddle/framework")
add_custom_target(framework_lib DEPENDS framework_py_proto
    COMMAND mkdir -p "${lib_dir}/details"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/framework/*.h" "${lib_dir}"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/framework/details/*.h" "${lib_dir}/details"
    COMMAND cp "${PADDLE_BINARY_DIR}/paddle/framework/framework.pb.h" "${lib_dir}"
)

set(lib_dir "${CMAKE_INSTALL_PREFIX}/paddle/memory")
add_custom_target(memory_lib
    COMMAND mkdir -p "${lib_dir}/detail"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/memory/*.h" "${lib_dir}"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/memory/detail/*.h" "${lib_dir}/detail"
)

set(lib_dir "${CMAKE_INSTALL_PREFIX}/paddle/inference")
add_custom_target(inference_lib DEPENDS paddle_fluid_shared
    COMMAND mkdir -p "${lib_dir}"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/inference/*.h" "${lib_dir}"
    COMMAND cp "${PADDLE_BINARY_DIR}/paddle/inference/libpaddle_fluid.so" "${lib_dir}"
)

set(lib_dir "${CMAKE_INSTALL_PREFIX}/paddle/platform")
add_custom_target(platform_lib
    COMMAND mkdir -p "${lib_dir}/dynload" "${lib_dir}/details"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/platform/*.h" "${lib_dir}"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/platform/dynload/*.h" "${lib_dir}/dynload"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/platform/details/*.h" "${lib_dir}/details"
)    

set(lib_dir "${CMAKE_INSTALL_PREFIX}/paddle/string")
add_custom_target(string_lib
    COMMAND mkdir -p "${lib_dir}/tinyformat"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/string/*.h" "${lib_dir}"
    COMMAND cp "${PADDLE_SOURCE_DIR}/paddle/string/tinyformat/*.h" "${lib_dir}/tinyformat"
)

add_custom_target(inference_lib_dist DEPENDS 
  inference_lib framework_lib memory_lib platform_lib string_lib
  gflags_lib glog_lib protobuf_lib eigen3_lib)
