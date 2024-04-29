file(
  GLOB spmd_rules_srcs
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")

collect_srcs(infermeta_srcs SRCS ${spmd_rules_srcs})
