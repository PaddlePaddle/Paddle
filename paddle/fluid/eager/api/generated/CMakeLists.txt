add_subdirectory(eager_generated)

if(NOT DEFINED PADDLE_ON_INFERENCE)
    add_subdirectory(fluid_generated)
endif()
