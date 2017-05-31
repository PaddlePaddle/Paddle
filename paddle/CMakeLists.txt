add_subdirectory(cuda)
add_subdirectory(function)
add_subdirectory(utils)
add_subdirectory(testing)
add_subdirectory(math)
add_subdirectory(parameter)
add_subdirectory(gserver)
add_subdirectory(pserver)
add_subdirectory(trainer)
add_subdirectory(scripts)

# Do not build go directory until go cmake is working smoothly.
# if(CMAKE_Go_COMPILER)
#   add_subdirectory(go)
# endif()

find_package(Boost QUIET)

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(majel)
endif()

if(WITH_C_API)
    add_subdirectory(capi)
endif()

if(WITH_SWIG_PY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
          ${CMAKE_CURRENT_SOURCE_DIR}/setup.py)
  add_subdirectory(api)
endif()
