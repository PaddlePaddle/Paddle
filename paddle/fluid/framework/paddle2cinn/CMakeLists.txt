file(GLOB paddle2cinn_srcs "*.cc")
set(paddle2cinn_deps
    absl
    isl
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
    tile_config_proto
    auto_schedule_proto
    common)
if(WITH_ONEDNN)
  set(paddle2cinn ${paddle2cinn} onednn)
endif()

if(WITH_TESTING)
  set(paddle2cinn_deps ${paddle2cinn_deps} gtest)
endif()

cc_library(
  paddle2cinn
  SRCS ${paddle2cinn_srcs}
  DEPS ${paddle2cinn_deps})
