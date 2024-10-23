file(
  GLOB reader_cc
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
collect_srcs(core_srcs SRCS ${reader_cc})
