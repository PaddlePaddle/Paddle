cc_library(mir_fusers
        SRCS conv_bn_fuser.cc
        DEPS pattern_matcher_high_api)

lite_cc_test(test_lite_conv_bn_fuse SRCS conv_bn_fuse_pass_test.cc
   DEPS elementwise_ops_lite batch_norm_op_lite conv_op_lite
proto_desc compatible_pb_lite program_lite mir_pass mir_pass_manager pattern_matcher_high_api)
