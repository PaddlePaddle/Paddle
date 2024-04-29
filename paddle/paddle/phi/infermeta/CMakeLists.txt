add_subdirectory(strings)
add_subdirectory(sparse)

if(WITH_DISTRIBUTE)
  add_subdirectory(spmd_rules)
endif()

collect_srcs(
  infermeta_srcs
  SRCS
  nullary.cc
  unary.cc
  binary.cc
  ternary.cc
  multiary.cc
  fusion.cc
  backward.cc)
