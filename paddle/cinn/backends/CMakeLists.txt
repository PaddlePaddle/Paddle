core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  outputs.cc
  codegen_c.cc
  codegen_c_x86.cc
  codegen_cuda_host.cc
  extern_func_emitter.cc
  extern_func_emitter_builtin.cc
  function_prototype.cc
  extern_func_protos.cc
  extern_func_jit_register.cc
  modular.cc
  compiler.cc)

if(WITH_CUDA)
  add_subdirectory(nvrtc)
  list(APPEND srcs cuda_util.cc codegen_cuda_dev.cc codegen_cuda_util.cc)
endif()

if(WITH_OPENMP)
  cinn_cc_library(__x86_source_fake_lib SRCS _x86_builtin_source.cc)
endif()
add_subdirectory(llvm)

if(WITH_CUDA)
  cinn_nv_test(test_raw_cuda_code SRCS raw_cuda_code_test.cu DEPS cinncore)
endif()

include_directories(${CMAKE_SOURCE_DIR}/paddle/cinn/runtime)

if(WITH_CUDA)
  cinn_nv_test(test_compiler SRCS compiler_test.cc DEPS cinncore)
else()
  cinn_cc_test(test_compiler SRCS compiler_test.cc DEPS cinncore)
endif()

foreach(cpp ${srcs})
  set(cinnapi_src
      "${cinnapi_src};paddle/cinn/backends/${cpp}"
      CACHE INTERNAL "")
endforeach()

file(
  GLOB includes
  LIST_DIRECTORIES false
  RELATIVE ${CMAKE_SOURCE_DIR}
  *.h)

foreach(header ${includes})
  set(core_includes
      "${core_includes};${header}"
      CACHE INTERNAL "")
endforeach()
