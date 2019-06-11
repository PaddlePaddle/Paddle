cc_library(mir_fuse_passes
        SRCS fc_fuse_pass.cc
        DEPS mir_pass mir_pass_manager pattern_matcher_high_api)
