gather_srcs(
  cluster_ops_src
  SRCS
  common_utils.cc
  shardable_axes_inferer.cc
  shardable_axes_provider.cc
  shardable_axes_utils.cc
  pattern_utils.cc
  fusion_helper.cc
  cluster_policy.cc
  clustering_engine.cc)

cc_library(cluster_ops SRCS ${cluster_ops_src})
