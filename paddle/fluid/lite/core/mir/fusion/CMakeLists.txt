lite_cc_library(fuse_fc
        SRCS fc_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_conv_elementwise_add_activation
        SRCS conv_elementwise_add_activation_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_conv_bn
        SRCS conv_bn_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_elementwise_add_activation
        SRCS elementwise_add_activation_fuser.cc
        DEPS pattern_matcher_high_api)
lite_cc_library(fuse_quant_dequant
        SRCS quant_dequant_op_fuser.cc
        DEPS pattern_matcher_high_api)

set(mir_fusers 
    fuse_fc 
    fuse_conv_elementwise_add_activation
    fuse_conv_bn
    fuse_quant_dequant
    fuse_elementwise_add_activation
    CACHE INTERNAL "fusers")

if (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()

lite_cc_test(test_lite_conv_bn_fuse SRCS conv_bn_fuse_pass_test.cc
   DEPS elementwise_ops_lite batch_norm_op_lite conv_op_lite proto_desc compatible_pb_lite program_lite mir_pass mir_pass_manager pattern_matcher_high_api)
