cc_library(stringpiece SRCS piece.cc)
cc_test(stringpiece_test SRCS piece_test.cc DEPS stringpiece glog gflags)
cc_test(stringprintf_test SRCS printf_test.cc DEPS glog gflags)
cc_test(to_string_test SRCS to_string_test.cc)

set(lib_dir "${CMAKE_INSTALL_PREFIX}/paddle/string")
add_custom_target(string_lib
    COMMAND mkdir -p "${lib_dir}/tinyformat"
    COMMAND cp "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${lib_dir}"
    COMMAND cp "${CMAKE_CURRENT_SOURCE_DIR}/tinyformat/*.h" "${lib_dir}/tinyformat"
)
