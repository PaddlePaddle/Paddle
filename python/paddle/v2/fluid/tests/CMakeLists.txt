file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(NOT WITH_DISTRIBUTE)
    list(REMOVE_ITEM TEST_OPS test_recv_op)
endif(NOT WITH_DISTRIBUTE)

list(REMOVE_ITEM TEST_OPS test_warpctc_op)
foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
py_test(test_warpctc_op SRCS test_warpctc_op.py ENVS FLAGS_warpctc_dir=${WARPCTC_LIB_DIR})

add_subdirectory(book)
add_subdirectory(book_distribute)
add_subdirectory(book_memory_optimization)
