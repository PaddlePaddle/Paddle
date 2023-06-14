if(NOT WITH_IR)
  return()
endif()

add_definitions(-DIR_DLL_EXPORT)

add_subdirectory(core)
add_subdirectory(pass)
add_subdirectory(pattern_rewrite)
