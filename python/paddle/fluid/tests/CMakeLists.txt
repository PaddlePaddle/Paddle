file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")


foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()

add_subdirectory(unittests)
add_subdirectory(book)

# TODO: support New Custom OP on Mac
if(Linux)
  add_subdirectory(custom_op)
endif()

# Windows CPU machine doesn't have CUDA, can't compile .cu file
# if(WIN32 AND WITH_GPU)
#   add_subdirectory(custom_op)
# endif()

set_tests_properties(test_beam_search_decoder PROPERTIES TIMEOUT 120)
