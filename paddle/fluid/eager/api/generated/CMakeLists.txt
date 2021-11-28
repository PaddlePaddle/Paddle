add_subdirectory(eager_generated)

if(NOT DEFINED ON_INFER)
    add_subdirectory(fluid_generated)
endif()
