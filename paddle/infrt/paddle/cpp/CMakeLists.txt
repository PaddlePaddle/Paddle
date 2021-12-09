core_gather_headers()

gather_srcs(infrt_src SRCS
    )

foreach(cpp ${SRCS})
  set(infrt_src
    "${infrt_src};infrt/paddle/cpp/${cpp}"
    CACHE INTERNAL "")
endforeach()

file(GLOB includes LIST_DIRECTORIES false RELATIVE ${CMAKE_SOURCE_DIR} *.h)

foreach(header ${includes})
  set(core_includes "${core_includes};${header}" CACHE INTERNAL "")
endforeach()
