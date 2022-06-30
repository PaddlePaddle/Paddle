if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  add_subdirectory(fluid_manual)
  set(fluid_manual_functions
      ${fluid_manual_functions}
      PARENT_SCOPE)
  set(fluid_manual_nodes
      ${fluid_manual_nodes}
      PARENT_SCOPE)
endif()
