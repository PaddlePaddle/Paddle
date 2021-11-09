include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/data/
    include(unity_build_rule.cmake)
endif()

cc_library(pipeline SRCS pipeline.cc DEPS parallel_executor simple_threadpool scope)
op_library(dataloader_op SRCS dataloader_op.cc dataloader_op.cu.cc DEPS pipeline ${OP_HEADER_DEPS})

# register_operators()

# TODO: add test here
# cc_test(xxx SRCS xxx DEPS xxx)
