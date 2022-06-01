cc_test(ivalue_test SRCS ivalue_test.cc DEPS phi phi_api_utils phi_tensor_raw)

if (WITH_TESTING AND NOT WIN32)
    add_custom_target(
        dy2st_download_program
        COMMAND wget -nc https://paddle-ci.gz.bcebos.com/dy2st/export.forward.pdiparams
        COMMAND wget -nc https://paddle-ci.gz.bcebos.com/dy2st/export.forward.pdmodel
        COMMAND wget -nc https://paddle-ci.gz.bcebos.com/dy2st/export.infer.pdiparams
        COMMAND wget -nc https://paddle-ci.gz.bcebos.com/dy2st/export.infer.pdmodel
    )
    set(JIT_DEPS executor parallel_executor executor_cache scope proto_desc phi phi_api_utils phi_tensor_raw
        elementwise_add_op matmul_v2_op activation_op reduce_mean_op feed_op fetch_op scale_op)
    cc_test(layer_test SRCS layer_test.cc DEPS ${JIT_DEPS})
    add_dependencies(layer_test dy2st_download_program)
endif()
