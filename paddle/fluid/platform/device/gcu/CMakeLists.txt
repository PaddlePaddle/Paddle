if(WITH_GCU)
  if(SANITIZER)
    if(${SANITIZER} IN_LIST "address;thread")
      add_compile_options(-fsanitize=${SANITIZER})
    endif()
  endif()

  add_compile_definitions(HLIR_BUILDER_ABI_COMPATIABLE)
  include_directories(/usr/include)
  include_directories(/usr/include/dtu)
  set(GCU_INFO_SRC "gcu_info.cc" "gcu_device.cc")
  message("GCU_INFO_SRC:" ${GCU_INFO_SRC})
  add_definitions(-DSUPPRESS_DEPRECATION_WARNING)
  cc_library(
    gcu_info
    SRCS ${GCU_INFO_SRC}
    DEPS enforce gcu_runtime)

  include_directories(./register)
  include_directories(./profile)
  include_directories(./utils)
  include_directories(./equivalence_trans)
  include_directories(./executor)
  include_directories(./common)

  file(GLOB UTILS_SRC ./utils/*)
  file(GLOB REGISTER_SRC ./register/*)
  file(GLOB_RECURSE TRNAS_SRC ./equivalence/*)
  file(GLOB COMPILER_SRC ./compiler/*)
  file(GLOB EXECUTOR_SRC ./executor/*.cc)
  file(GLOB COMMON_SRC ./common/*.cc)
  set(BACKEND_SRC "gcu_backend.cc" "gcu_strategy.cc" "gcu_compiler.cc")

  set(TOTAL_BACKEND_SRC
      ${BACKEND_SRC}
      ${UTILS_SRC}
      ${REGISTER_SRC}
      ${TRNAS_SRC}
      ${COMPILER_SRC}
      ${EXECUTOR_SRC}
      ${COMMON_SRC})

  message("GCU_BACKEND_SRC:" ${TOTAL_BACKEND_SRC})

  add_subdirectory(runtime)
  add_subdirectory(layout)
  add_subdirectory(compiler)

  cc_library(
    gcu_backend
    SRCS ${TOTAL_BACKEND_SRC}
    DEPS graph dtu_sdk enforce process_group gcu_info)
endif()
