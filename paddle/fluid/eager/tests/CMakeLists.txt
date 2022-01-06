add_subdirectory(data_structure_tests)
add_subdirectory(task_tests)

if(NOT ON_INFER)
    add_subdirectory(performance_tests)
endif()
