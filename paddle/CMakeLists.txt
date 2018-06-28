if(NOT WITH_FLUID_ONLY)
  add_subdirectory(cuda)
  add_subdirectory(function)
  add_subdirectory(utils)
  add_subdirectory(math)
  add_subdirectory(gserver)
  add_subdirectory(parameter)

  if(MOBILE_INFERENCE)
    add_subdirectory(capi)
  else()
    add_subdirectory(pserver)
    add_subdirectory(trainer)
    add_subdirectory(scripts)

# Disable the build of C-API, which is the inference library of old PaddlePaddle.
# I want to disable it because the capi/CMakeLists.txt depends on SWIG, which 
# is no longer useful with Fluid and would bother the development of Tape.
# Once this removal is confirmed alright by the fact that no one askes the C-API
# in a while, we would be confident to clear the code.
#    if(WITH_C_API)
#      add_subdirectory(capi)
#    endif()

    if(WITH_SWIG_PY)
      add_subdirectory(api)
    endif()
  endif()
endif()

add_subdirectory(testing)
if(NOT MOBILE_INFERENCE AND NOT RPI AND NOT WITH_C_API)
  add_subdirectory(fluid)
endif()
