set(srcs var_desc.cc op_desc.cc block_desc.cc program_desc.cc)

foreach(cpp ${srcs})
  set(cinnapi_src
      "${cinnapi_src};paddle/cinn/frontend/paddle/cpp/${cpp}"
      CACHE INTERNAL "")
endforeach()

file(
  GLOB includes
  LIST_DIRECTORIES false
  RELATIVE ${CMAKE_SOURCE_DIR}
  *.h)

foreach(header ${includes})
  set(core_includes
      "${core_includes};${header}"
      CACHE INTERNAL "")
endforeach()
