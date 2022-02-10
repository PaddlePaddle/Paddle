gather_srcs(infrt_src SRCS
    var_desc.cc
    op_desc.cc
    block_desc.cc
    program_desc.cc
    )

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
