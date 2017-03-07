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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/setup.py)

if(WITH_PREDICT_SDK)
    add_subdirectory(predict)
endif()

if(WITH_SWIG_PY)
  add_subdirectory(api)
endif()
