add_subdirectory(frontend)
add_subdirectory(backend)
add_subdirectory(policy)
add_subdirectory(pir_graph_analyzing)

cc_library(
  op_fusion
  SRCS ${frontend_fusion_src} ${backend_fusion_src} ${policy_fusion_src}
       ${fusion_graph_analyzing} pattern_graph.cc
  DEPS phi)
