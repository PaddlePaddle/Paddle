set(srcs
    runtime.cc
    common.cc
    lang.cc
    ir/ir.cc
    ir/ir_api.cc
    ir/ir_context.cc
    poly.cc
    backends.cc
    bind.cc
    optim.cc
    pe.cc
    framework.cc
    utils.cc
    schedule.cc)

gather_srcs(cinnapi_src SRCS ${srcs})

if(WITH_CUDA)
  message(STATUS "Compile core_api with CUDA support")
  cinn_nv_library(
    core_api
    SHARED
    SRCS
    ${srcs}
    DEPS
    cinncore_static
    cinn_runtime
    pybind
    common)
  message("cuda_nvrtc: ${CUDA_NVRTC}")
  target_link_libraries(core_api ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES} cuda cudnn)
  if(NVTX_FOUND)
    target_link_libraries(core_api ${CUDA_NVTX_LIB})
  endif()
else()
  message(STATUS "Compile core_api without CUDA support")
  cinn_cc_library(
    core_api
    SHARED
    SRCS
    ${srcs}
    DEPS
    cinncore_static
    cinn_runtime
    pybind
    ${llvm_libs})
endif()

target_link_libraries(core_api ${MKLML_LIB} isl ginac common)
if(USE_OPENMP STREQUAL "gnu")
  target_link_libraries(core_api ${OpenMP_CXX_LIBRARIES})
  message(STATUS "OpenMP lib: ${OpenMP_CXX_LIBRARIES}")
elseif(USE_OPENMP STREQUAL "intel")
  target_link_libraries(core_api ${MKLML_IOMP_LIB})
  message(STATUS "OpenMP lib: ${MKLML_IOMP_LIB}")
endif()

set_target_properties(core_api PROPERTIES PREFIX "")
