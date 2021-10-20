include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/data/
    include(unity_build_rule.cmake)
endif()

cc_library(pipeline SRCS pipeline.cc DEPS parallel_executor simple_threadpool)
op_library(dataloader_op SRCS dataloader_op.cc DEPS pipeline)

# register_operators()

# TODO: add test here
# cc_test(xxx SRCS xxx DEPS xxx)
