if (NOT LITE_SUBGRAPH_PATH)
    message(STATUS "==== No LITE_SUBGRAPH! ====")
    return()
else()
    # Print path of subgraph.
    message(STATUS "==== Build LITE_SUBGRAPH! ====")
    message(STATUS "LITE_SUBGRAPH_PATH: ${LITE_SUBGRAPH_PATH}")

    # Set the LITE_SUBGRAPH_LIB_PATH variable.
    find_library(LITE_SUBGRAPH_LIB_PATH NAMES libapi_full_static.a
        PATHS ${LITE_SUBGRAPH_PATH}
        NO_DEFAULT_PATH
        DOC "Path to lite library.")

    # Add library api_full_static by path LITE_SUBGRAPH_LIB_PATH.
    add_library(api_full_static STATIC IMPORTED GLOBAL)
    SET_PROPERTY(TARGET api_full_static PROPERTY IMPORTED_LOCATION ${LITE_SUBGRAPH_LIB_PATH})

    # Make target model_optimize_tool. The sequence of glog gflags needs to be attention.
    message(STATUS "LITE_SUBGRAPH_LIB_PATH: ${LITE_SUBGRAPH_LIB_PATH}")
    include_directories(${LITE_SUBGRAPH_PATH}/include)
    cc_binary(model_optimize_tool SRCS model_optimize_tool.cc DEPS api_full_static glog gflags)

endif()