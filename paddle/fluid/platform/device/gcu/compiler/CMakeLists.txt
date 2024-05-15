if(WITH_GCU)
  add_compile_definitions(HLIR_BUILDER_ABI_COMPATIABLE)
  include_directories(./)
  include_directories(/usr/include)
  include_directories(/usr/include/dtu)
  include_directories(/usr/include/tops)

  file(GLOB GCU_COMPILER_SRC ./*.cc)

  cc_library(
    gcu_compiler
    SRCS ${GCU_COMPILER_SRC}
    DEPS gcu_runtime enforce graph)

endif()
