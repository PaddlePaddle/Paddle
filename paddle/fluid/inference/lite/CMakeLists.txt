if (LITE_SUBGRAPH_PATH)
    # Print path of subgraph.
    message(STATUS "==== Build LITE_SUBGRAPH! ====")
    message(STATUS "LITE_SUBGRAPH_PATH: ${LITE_SUBGRAPH_PATH}")

    # Set the LITE_SUBGRAPH_LIB_PATH variable.
    find_library(LITE_SUBGRAPH_LIB_PATH NAMES liblite_full_static.a
        PATHS ${LITE_SUBGRAPH_PATH}
        NO_DEFAULT_PATH
        DOC "Path to lite library.")

    # Add library lite_full_static by path LITE_SUBGRAPH_LIB_PATH.
    add_library(lite_full_static STATIC IMPORTED GLOBAL)
    SET_PROPERTY(TARGET lite_full_static PROPERTY IMPORTED_LOCATION ${LITE_SUBGRAPH_LIB_PATH})

    # Make target model_optimize_tool. The sequence of glog gflags needs to be attention.
    message(STATUS "LITE_SUBGRAPH_LIB_PATH: ${LITE_SUBGRAPH_LIB_PATH}")
    include_directories(${LITE_SUBGRAPH_PATH}/include)
endif()

cc_binary(test_leaky_relu SRCS test_leaky_relu.cc DEPS lite_full_static dynload_cuda glog gflags gtest)
cc_library(lite_op_teller SRCS op_teller.cc DEPS framework_proto device_context boost xxhash)
cc_library(lite_engine SRCS engine.cc DEPS lite_full_static framework_proto)
cc_library(lite_tensor_utils SRCS tensor_utils.cc DEPS lite_full_static framework_proto boost)
