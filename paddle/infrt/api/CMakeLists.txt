core_gather_headers()

gather_srcs(infrt_src SRCS
    infrt_api.cc
    )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/infrt_api_test.cc.in ${CMAKE_CURRENT_SOURCE_DIR}/infrt_api_test.cc)

# Disable temporarily for the external-kernel's mkldnn is outdate
cc_test_tiny(test_infrt_api SRCS infrt_api_test.cc DEPS infrt ${MLIR_IR_LIBS})
# TODO(inference): remove after optimize weight unfold.
set_tests_properties(test_infrt_api PROPERTIES TIMEOUT 200)
