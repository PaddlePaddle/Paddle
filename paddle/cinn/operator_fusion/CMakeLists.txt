add_subdirectory(frontend)
add_subdirectory(backend)
add_subdirectory(policy)

cc_library(
  op_fusion
  SRCS ${frontend_fusion_src} ${backend_fusion_src} ${policy_fusion_src}
       pattern_graph.cc
  DEPS phi)
