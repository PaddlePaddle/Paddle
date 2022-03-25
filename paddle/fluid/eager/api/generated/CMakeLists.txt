add_subdirectory(eager_generated)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    add_subdirectory(fluid_generated)
endif()
