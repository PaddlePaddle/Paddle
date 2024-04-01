gather_srcs(group_cluster_src SRCS common_utils.cc pattern_node.cc
            pattern_graph.cc)

add_subdirectory(cluster_policy)

cc_library(group_cluster SRCS ${group_cluster_src})
