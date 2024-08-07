add_subdirectory(policy)
add_subdirectory(pir_graph_analyzing)
add_subdirectory(fusion_tracker)

cc_library(
  op_fusion
  SRCS ${policy_fusion_src} ${fusion_graph_analyzing} ${fusion_tracker_src}
       pattern_graph.cc utils.cc fusion_interface.cc
  DEPS phi)
