cc_library(fuse_fc
        SRCS fc_fuser.cc
        DEPS pattern_matcher_high_api)
cc_library(conv_elementwise_add_relu
        SRCS conv_elementwise_add_relu_fuser.cc
        DEPS pattern_matcher_high_api)

set(mir_fusers 
    fuse_fc 
    conv_elementwise_add_relu
    CACHE INTERNAL "fusers")
