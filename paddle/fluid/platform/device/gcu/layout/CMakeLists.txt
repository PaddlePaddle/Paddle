if(WITH_GCU)
  add_compile_definitions(HLIR_BUILDER_ABI_COMPATIABLE)
  include_directories(./)
  include_directories(/usr/include)
  include_directories(/usr/include/dtu)
  include_directories(/usr/include/tops)

  file(GLOB GCU_LAYOUT_SRC ./gcu_layout_interface.cc)
  message(wxl debug)
  message(${GCU_LAYOUT_SRC})

  cc_library(
    gcu_layout
    SRCS ${GCU_LAYOUT_SRC}
    DEPS gcu_runtime enforce gcu_compiler)

endif()
