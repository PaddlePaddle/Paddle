cc_library(fuse_fc
        SRCS fc_fuser.cc
        DEPS pattern_matcher_high_api)
cc_library(fuse_conv_elementwise_add_relu
        SRCS conv_elementwise_add_relu_fuser.cc
        DEPS pattern_matcher_high_api)
cc_library(fuse_conv_bn
        SRCS conv_bn_fuser.cc
        DEPS pattern_matcher_high_api)

set(mir_fusers 
    fuse_fc 
    fuse_conv_elementwise_add_relu
    fuse_conv_bn
    CACHE INTERNAL "fusers")

lite_cc_test(test_lite_conv_bn_fuse SRCS conv_bn_fuse_pass_test.cc
   DEPS elementwise_ops_lite batch_norm_op_lite conv_op_lite proto_desc compatible_pb_lite program_lite mir_pass mir_pass_manager pattern_matcher_high_api)
