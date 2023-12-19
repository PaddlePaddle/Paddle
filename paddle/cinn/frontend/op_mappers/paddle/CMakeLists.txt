core_gather_headers()
file(
  GLOB paddle_op_mapper_srcs
  LIST_DIRECTORIES false
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  *.cc)
gather_srcs(cinnapi_src SRCS ${paddle_op_mapper_srcs})
