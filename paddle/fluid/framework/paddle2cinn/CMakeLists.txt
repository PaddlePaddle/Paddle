file(GLOB paddle2cinn_srcs "*.cc")
set(paddle2cinn_deps
    gtest
    absl
    isl
    mkldnn
    xxhash
    pybind
    python
    op_registry
    pass
    graph_traits
    cinnapi
    cinn_launch_context
    proto_desc
    auto_parallel_proto
    cinn_framework_proto
    schedule_desc_proto
    auto_schedule_proto
    parallel_executor)

cc_library(
  paddle2cinn
  SRCS ${paddle2cinn_srcs}
  DEPS ${paddle2cinn_deps})
