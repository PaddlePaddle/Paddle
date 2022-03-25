add_subdirectory(data_structure_tests)
add_subdirectory(task_tests)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    add_subdirectory(performance_tests)
endif()
