set(op_DEPS ${tensor_lite} op_lite op_params_lite)

lite_cc_library(conv_op_lite SRCS conv_op.cc DEPS ${op_DEPS})
lite_cc_library(pool_op_lite SRCS pool_op.cc DEPS ${op_DEPS})
lite_cc_library(fc_op_lite SRCS fc_op.cc DEPS ${op_DEPS})
lite_cc_library(relu_op_lite SRCS relu_op.cc DEPS ${op_DEPS})
lite_cc_library(mul_op_lite SRCS mul_op.cc DEPS ${op_DEPS})
lite_cc_library(scale_op_lite SRCS scale_op.cc DEPS ${op_DEPS})
lite_cc_library(softmax_op_lite SRCS softmax_op.cc DEPS ${op_DEPS})
lite_cc_library(reshape_op_lite SRCS reshape_op.cc DEPS ${op_DEPS} )
lite_cc_library(batch_norm_op_lite SRCS batch_norm_op.cc DEPS ${op_DEPS})
lite_cc_library(feed_op_lite SRCS feed_op.cc DEPS ${op_DEPS})
lite_cc_library(fetch_op_lite SRCS fetch_op.cc DEPS ${op_DEPS})
lite_cc_library(io_copy_op_lite SRCS io_copy_op.cc DEPS ${op_DEPS})
lite_cc_library(activation_ops_lite SRCS activation_ops.cc DEPS ${op_DEPS})
lite_cc_library(elementwise_ops_lite SRCS elementwise_ops.cc DEPS ${op_DEPS})
lite_cc_library(fusion_elementwise_activation_ops_lite SRCS fusion_elementwise_activation_ops.cc DEPS elementwise_ops_lite ${op_DEPS})
lite_cc_library(mean_op_lite SRCS mean_op.cc DEPS ${op_DEPS})
lite_cc_library(fill_constant_op_lite SRCS fill_constant_op.cc DEPS ${op_DEPS})
lite_cc_library(sgd_op_lite SRCS sgd_op.cc DEPS ${op_DEPS})
lite_cc_library(uniform_random_op_lite SRCS uniform_random_op.cc DEPS ${op_DEPS})

lite_cc_library(op_params_lite SRCS op_params.cc DEPS ${tensor_lite} any_lite framework_proto_lite)
lite_cc_library(dropout_op_lite SRCS dropout_op.cc DEPS ${op_DEPS})
lite_cc_library(concat_op_lite SRCS concat_op.cc DEPS ${op_DEPS})
lite_cc_library(calib_op_lite SRCS calib_op.cc DEPS ${op_DEPS})
lite_cc_library(split_op_lite SRCS split_op.cc DEPS ${op_DEPS})
lite_cc_library(transpose_op_lite SRCS transpose_op.cc DEPS ${op_DEPS})
lite_cc_library(fake_quant SRCS fake_quantize_moving_avg_max_abs.cc DEPS ${op_DEPS})
lite_cc_library(fake_dequant SRCS fake_dequantize_max_abs.cc DEPS ${op_DEPS})

set(ops_lite
        conv_op_lite
        pool_op_lite
        fc_op_lite
        relu_op_lite
        mul_op_lite
        scale_op_lite
        softmax_op_lite
        reshape_op_lite
        batch_norm_op_lite
        feed_op_lite
        fetch_op_lite
        io_copy_op_lite
        elementwise_ops_lite
        fusion_elementwise_activation_ops_lite
        mean_op_lite
        fill_constant_op_lite
        activation_ops_lite
        dropout_op_lite
        concat_op_lite
        calib_op_lite
        split_op_lite
        transpose_op_lite
        fake_quant
        fake_dequant
        sgd_op_lite
        uniform_random_op_lite
        CACHE INTERNAL "ops lite")

lite_cc_test(test_fc_op_lite SRCS fc_op_test.cc 
             DEPS fc_op_lite memory_lite 
             X86_DEPS fc_compute_x86
             ARM_DEPS fc_compute_arm)
lite_cc_test(test_pool_op_lite SRCS pool_op_test.cc
             DEPS pool_op_lite memory_lite
             ARM_DEPS pool_compute_arm)
lite_cc_test(test_scale_op_lite SRCS scale_op_test.cc DEPS scale_op_lite memory_lite)
lite_cc_test(test_softmax_op_lite SRCS softmax_op_test.cc DEPS softmax_op_lite memory_lite)
lite_cc_test(test_reshape_op_lite SRCS reshape_op_test.cc DEPS reshape_op_lite memory_lite)
lite_cc_test(test_batch_norm_op_lite SRCS batch_norm_op_test.cc DEPS batch_norm_op_lite memory_lite)
lite_cc_test(test_concat_op_lite SRCS concat_op_test.cc DEPS concat_op_lite memory_lite)
lite_cc_test(test_calib_op_lite SRCS calib_op_test.cc DEPS calib_op_lite memory_lite ARM_DEPS calib_compute_arm)
lite_cc_test(test_fusion_elementwise_activation_ops_lite 
             SRCS fusion_elementwise_activation_ops_test.cc 
             DEPS fusion_elementwise_activation_ops_lite memory_lite)
lite_cc_test(test_transpose_op_lite SRCS transpose_op_test.cc DEPS transpose_op_lite memory_lite)
