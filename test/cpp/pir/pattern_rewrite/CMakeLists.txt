paddle_test(pattern_rewrite_test SRCS pattern_rewrite_test.cc)

paddle_test(drr_test SRCS drr_test.cc)

paddle_test(drr_same_type_binding_test SRCS drr_same_type_binding_test.cc)

paddle_test(drr_fuse_linear_test SRCS drr_fuse_linear_test.cc)

paddle_test(drr_fuse_linear_param_grad_add_test SRCS
            drr_fuse_linear_param_grad_add_test.cc)

paddle_test(drr_attention_fuse_test SRCS drr_attention_fuse_test.cc)
