if(WITH_IPU)
  set(paddle_ipu_handler ${CMAKE_CURRENT_BINARY_DIR}/paddle_ipu_handler.h.tmp)
  set(paddle_ipu_handler_final ${CMAKE_CURRENT_BINARY_DIR}/paddle_ipu_handler.h)
  file(WRITE ${paddle_ipu_handler}
       "// Auto generated from CMake. DO NOT EDIT!\n\n")
  file(APPEND ${paddle_ipu_handler} "\#pragma once\n")
  file(
    APPEND ${paddle_ipu_handler}
    "\#include \"paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h\"\n\n"
  )
  file(GLOB POPART_CANONICALIZATION_SRC
       ${CMAKE_CURRENT_SOURCE_DIR}/popart_canonicalization/*.cc)
  copy_if_different(${paddle_ipu_handler} ${paddle_ipu_handler_final})

  foreach(file_path ${POPART_CANONICALIZATION_SRC})
    file(READ ${file_path} file_content)
    string(REGEX MATCHALL "(REGISTER_HANDLER)(\\()([A-Za-z0-9_]+)(,)"
                 op_handlers ${file_content})
    string(REPLACE "REGISTER_HANDLER(" "" op_handlers "${op_handlers}")
    string(REPLACE "," "" op_handlers "${op_handlers}")
    foreach(op_handler ${op_handlers})
      file(APPEND ${paddle_ipu_handler} "USE_HANDLER(${op_handler});\n")
    endforeach()
  endforeach()

<<<<<<< HEAD
  set(IPU_BACKEND_SRC "ipu_strategy.cc" "ipu_executor.cc" "ipu_compiler.cc"
=======
  set(IPU_BACKEND_SRC "ipu_strategy.cc" "ipu_compiler.cc" "ipu_executor.cc"
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                      "ipu_backend.cc" "ipu_utils.cc")
  set(IPU_INFO_SRC "ipu_info.cc" "ipu_device.cc")

  cc_library(
    popart_canonicalization
    SRCS ${POPART_CANONICALIZATION_SRC}
    DEPS graph)
  cc_library(
    ipu_backend
    SRCS ${IPU_BACKEND_SRC}
<<<<<<< HEAD
    DEPS popart-only graph graph_helper popdist popart_canonicalization)
=======
    DEPS popart-only
         graph
         graph_helper
         popdist
         popef
         model_runtime
         popart_canonicalization)

  # magic here is model_runtime requires CXX17 and uses std::optional while popart uses CXX14
  # and nonstd::optional. A manually macro setting is required here to solve symbol conflict.
  target_compile_features(ipu_backend PRIVATE cxx_std_17)
  target_compile_definitions(ipu_backend
                             PRIVATE optional_CONFIG_SELECT_OPTIONAL=1)

>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  cc_library(
    ipu_info
    SRCS ${IPU_INFO_SRC}
    DEPS popart-only enforce)
endif()
