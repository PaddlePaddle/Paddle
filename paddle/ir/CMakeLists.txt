if(NOT WITH_IR)
  return()
endif()

add_subdirectory(core)
add_subdirectory(pass)
add_subdirectory(pattern_rewrite)
