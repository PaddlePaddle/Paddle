set(PYBIND_DEPS
    init
    pybind
    python
    proto_desc
    memory
    executor
    fleet_wrapper
    box_wrapper
    metrics
    prune
    feed_fetch_method
    pass
    generate_pass
    pass_builder
    parallel_executor
    profiler
    layer
    tracer
    engine
    scope_pool
    analysis_predictor
    imperative_profiler
    imperative_flag
    save_load_util
    dlpack_tensor
    device_context
    gloo_wrapper
    infer_io_utils
    heter_wrapper
    generator
    op_version_registry
    ps_gpu_wrapper
    custom_operator
    cost_model
    cuda_graph_with_memory_pool
    fleet_executor
    global_utils
    phi_utils
    tcp_store
    new_profiler
    auto_parallel
    jit_layer
    jit_property)

if(WITH_PSCORE)
  set(PYBIND_DEPS ${PYBIND_DEPS} ps_service)
  if(WITH_HETERPS)
    set(PYBIND_DEPS ${PYBIND_DEPS} graph_gpu_wrapper)
  endif()
endif()
if(WITH_GPU OR WITH_ROCM)
  set(PYBIND_DEPS ${PYBIND_DEPS} dynload_cuda)
  set(PYBIND_DEPS ${PYBIND_DEPS} cuda_device_guard)
endif()

if(WITH_GPU)
  set(PYBIND_DEPS ${PYBIND_DEPS} cuda_profiler)
endif()
if(WITH_IPU)
  set(PYBIND_DEPS ${PYBIND_DEPS} ipu_info)
endif()

if(WITH_NCCL OR WITH_RCCL)
  set(PYBIND_DEPS ${PYBIND_DEPS} nccl_wrapper)
  set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
endif()

if(WITH_XPU_BKCL)
  set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
  set(PYBIND_DEPS ${PYBIND_DEPS} bkcl_context)
  set(PYBIND_DEPS ${PYBIND_DEPS} heter_ccl_context)
endif()

if(WITH_ASCEND_CL)
  set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
  set(PYBIND_DEPS ${PYBIND_DEPS} hccl_context)
  set(PYBIND_DEPS ${PYBIND_DEPS} heter_ccl_context)
endif()

if(WITH_CNCL)
  set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
  set(PYBIND_DEPS ${PYBIND_DEPS} cncl_context)
endif()

if(NOT WIN32)
  set(PYBIND_DEPS ${PYBIND_DEPS} data_loader)
  if(WITH_NCCL OR WITH_RCCL)
    set(PYBIND_DEPS ${PYBIND_DEPS} nccl_context)
    set(PYBIND_DEPS ${PYBIND_DEPS} heter_ccl_context)
  endif()
endif()

if(WITH_PYTHON)
  list(APPEND PYBIND_DEPS py_func_op)
  list(APPEND PYBIND_DEPS py_layer_op)
endif()

set(PYBIND_SRCS
    pybind.cc
    imperative.cc
    inference_api.cc
    ir.cc
    bind_fleet_executor.cc
    reader_py.cc
    protobuf.cc
    exception.cc
    op_function_common.cc
    parallel_executor.cc
    tensor.cc
    place.cc
    const_value.cc
    global_value_getter_setter.cc
    fleet_wrapper_py.cc
    heter_wrapper_py.cc
    ps_gpu_wrapper_py.cc
    gloo_wrapper_py.cc
    box_helper_py.cc
    metrics_py.cc
    data_set_py.cc
    bind_cost_model.cc
    compatible.cc
    io.cc
    generator_py.cc
    communication.cc
    cuda_streams_py.cc
    jit.cc
    auto_parallel_py.cc
    op_function1.cc
    op_function2.cc
    op_function3.cc
    op_function4.cc
    op_function5.cc
    op_function6.cc
    op_function7.cc
    op_function8.cc)

if(WITH_CUSTOM_DEVICE)
  set(PYBIND_DEPS ${PYBIND_DEPS} phi_capi)
  set(PYBIND_DEPS ${PYBIND_DEPS} custom_device_common_op_registry)
endif()

if(NOT ON_INFER)
  set(PYBIND_DEPS ${PYBIND_DEPS} processgroup eager_reducer)
  if(WITH_NCCL OR WITH_RCCL)
    set(PYBIND_DEPS ${PYBIND_DEPS} processgroup_nccl)
    if(WITH_PSCORE)
      set(PYBIND_DEPS ${PYBIND_DEPS} processgroup_heter)
    endif()
  endif()
  if(WITH_GLOO)
    set(PYBIND_DEPS ${PYBIND_DEPS} processgroup_gloo)
  endif()
  if(WITH_ASCEND_CL)
    set(PYBIND_DEPS ${PYBIND_DEPS} processgroup_hccl)
    if(WITH_PSCORE)
      set(PYBIND_DEPS ${PYBIND_DEPS} processgroup_heter)
    endif()
  endif()
  if(WITH_CUSTOM_DEVICE)
    set(PYBIND_DEPS ${PYBIND_DEPS} processgroup_custom)
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
    set_source_files_properties(
      distributed_py.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  endif()
  set(PYBIND_SRCS ${PYBIND_SRCS} distributed_py.cc)
endif()

if(WITH_ASCEND)
  set(PYBIND_DEPS ${PYBIND_DEPS} ascend_wrapper)
  set(PYBIND_SRCS ${PYBIND_SRCS} ascend_wrapper_py.cc)
endif()

if(WITH_GLOO)
  set(PYBIND_DEPS ${PYBIND_DEPS} gloo_context)
  set(PYBIND_SRCS ${PYBIND_SRCS} gloo_context_py.cc)
  set(PYBIND_DEPS ${PYBIND_DEPS} imperative_gloo_context)
  set(PYBIND_DEPS ${PYBIND_DEPS} reducer)
endif()

if(WITH_CRYPTO)
  set(PYBIND_DEPS ${PYBIND_DEPS} paddle_crypto)
  set(PYBIND_SRCS ${PYBIND_SRCS} crypto.cc)
endif()

if(WITH_PSLIB)
  set(DISTRIBUTE_COMPILE_FLAGS
      "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=sign-compare -Wno-error=unused-variable -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=type-limits -Wno-error=unknown-pragmas -Wno-error=parentheses -Wno-error=unused-result"
  )
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
  endif()
  set_source_files_properties(
    heter_wrapper_py.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endif()
if(WITH_PSCORE)
  if(WITH_ARM_BRPC)
    set(DISTRIBUTE_COMPILE_FLAGS
        "-faligned-new -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=sign-compare -Wno-error=unused-variable -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=unknown-pragmas -Wno-error=parentheses -Wno-error=unused-result"
    )
  else()
    set(DISTRIBUTE_COMPILE_FLAGS
        "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=sign-compare -Wno-error=unused-variable -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=unknown-pragmas -Wno-error=parentheses -Wno-error=unused-result"
    )
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
      set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
    endif()
  endif()
  set_source_files_properties(
    fleet_py.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
  list(APPEND PYBIND_DEPS fleet index_wrapper index_sampler)
  list(APPEND PYBIND_SRCS)
  set(PYBIND_SRCS fleet_py.cc ${PYBIND_SRCS})
endif()

if(WITH_NCCL OR WITH_RCCL)
  list(APPEND PYBIND_SRCS nccl_wrapper_py.cc)
endif()

if(WITH_PYTHON)
  # generate op pybind functions automatically for dygraph.
  if(WITH_ASCEND_CL)
    set(OP_FUNCTION_GENERETOR_DEPS
        pybind
        proto_desc
        executor
        layer
        tracer
        engine
        imperative_profiler
        imperative_flag
        ascend_wrapper)
  else()
    set(OP_FUNCTION_GENERETOR_DEPS
        pybind
        proto_desc
        executor
        layer
        tracer
        engine
        imperative_profiler
        imperative_flag)
  endif()
  list(APPEND OP_FUNCTION_GENERETOR_DEPS ${GLOB_OP_LIB})
  list(APPEND OP_FUNCTION_GENERETOR_DEPS ${GLOB_OPERATOR_DEPS})

  if(WITH_NCCL OR WITH_RCCL)
    list(APPEND OP_FUNCTION_GENERETOR_DEPS nccl_context)
  endif()

  if(WITH_XPU_BKCL)
    list(APPEND OP_FUNCTION_GENERETOR_DEPS bkcl_context)
  endif()

  if(WITH_ASCEND_CL)
    list(APPEND OP_FUNCTION_GENERETOR_DEPS hccl_context)
  endif()

  if(WITH_CNCL)
    list(APPEND OP_FUNCTION_GENERETOR_DEPS cncl_context)
  endif()

  add_executable(op_function_generator op_function_generator.cc)
  target_link_libraries(op_function_generator ${OP_FUNCTION_GENERETOR_DEPS})
  add_executable(eager_legacy_op_function_generator
                 eager_legacy_op_function_generator.cc)
  target_link_libraries(eager_legacy_op_function_generator
                        ${OP_FUNCTION_GENERETOR_DEPS})
  if(NOT WIN32)
    add_executable(kernel_signature_generator kernel_signature_generator.cc)
    target_link_libraries(kernel_signature_generator
                          ${OP_FUNCTION_GENERETOR_DEPS})
  endif()

  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(op_function_generator ${os_dependency_modules})
  target_link_libraries(eager_legacy_op_function_generator
                        ${os_dependency_modules})
  if(WITH_ROCM)
    target_link_libraries(op_function_generator ${ROCM_HIPRTC_LIB})
    target_link_libraries(eager_legacy_op_function_generator ${ROCM_HIPRTC_LIB})
    target_link_libraries(kernel_signature_generator ${ROCM_HIPRTC_LIB})
  endif()

  set(op_function_output_path ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/)
  set(impl_file1 ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function1.cc)
  set(tmp_impl_file1 ${impl_file1}.tmp)
  set(impl_file2 ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function2.cc)
  set(tmp_impl_file2 ${impl_file2}.tmp)
  set(impl_file3 ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function3.cc)
  set(tmp_impl_file3 ${impl_file3}.tmp)
  set(impl_file4 ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function4.cc)
  set(tmp_impl_file4 ${impl_file4}.tmp)
  set(impl_file5 ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function5.cc)
  set(tmp_impl_file5 ${impl_file5}.tmp)
  set(impl_file6 ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function6.cc)
  set(tmp_impl_file6 ${impl_file6}.tmp)
  set(impl_file7 ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function7.cc)
  set(tmp_impl_file7 ${impl_file7}.tmp)
  set(impl_file8 ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/op_function8.cc)
  set(tmp_impl_file8 ${impl_file8}.tmp)
  set(CODE_GEN_SPLIT_FILE_COUNT "8")
  set(eager_impl_file
      ${CMAKE_SOURCE_DIR}/paddle/fluid/pybind/eager_legacy_op_function.cc)
  set(tmp_eager_impl_file ${eager_impl_file}.tmp)

  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${PADDLE_SOURCE_DIR}/paddle/fluid/pybind/generate_file_structures.py"
      "${PADDLE_SOURCE_DIR}/paddle/fluid/pybind/"
      "${CODE_GEN_SPLIT_FILE_COUNT}")

  set(OP_IMPL_DEPS op_function_generator)
  set(EAGER_OP_IMPL_DEPS eager_legacy_op_function_generator
                         eager_python_c_codegen)

  if(WIN32)
    if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
      set(op_impl_path "${CMAKE_CURRENT_BINARY_DIR}")
    else()
      set(op_impl_path "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
    endif()

    file(
      WRITE
      ${CMAKE_BINARY_DIR}/paddle/fluid/pybind/op_function_generator_retry.bat
      ""
      "set build_times=1\n"
      ":retry\n"
      "ECHO op_function_generator run %build_times% time\n"
      "taskkill /f /im op_function_generator.exe 2>NUL\n"
      "${op_impl_path}/op_function_generator.exe ${op_function_output_path} ${CODE_GEN_SPLIT_FILE_COUNT}\n"
      "if %ERRORLEVEL% NEQ 0 (\n"
      "    set /a build_times=%build_times%+1\n"
      "    if %build_times% GEQ 10 (\n"
      "        exit /b 1\n"
      "    ) else (\n"
      "        goto :retry\n"
      "    )\n"
      ")\n"
      "exit /b 0")

    file(
      WRITE
      ${CMAKE_BINARY_DIR}/paddle/fluid/pybind/eager_legacy_op_function_generator_retry.bat
      ""
      "set build_times=1\n"
      ":retry\n"
      "ECHO eager_legacy_op_function_generator run %build_times% time\n"
      "taskkill /f /im eager_legacy_op_function_generator.exe 2>NUL\n"
      "${op_impl_path}/eager_legacy_op_function_generator.exe ${tmp_eager_impl_file}\n"
      "if %ERRORLEVEL% NEQ 0 (\n"
      "    set /a build_times=%build_times%+1\n"
      "    if %build_times% GEQ 10 (\n"
      "        exit /b 1\n"
      "    ) else (\n"
      "        goto :retry\n"
      "    )\n"
      ")\n"
      "exit /b 0")

    if(${CBLAS_PROVIDER} STREQUAL MKLML)
      add_custom_command(
        OUTPUT ${op_impl_path}/libiomp5md.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_IOMP_LIB}
                ${op_impl_path}
        DEPENDS mklml)
      list(APPEND OP_IMPL_DEPS ${op_impl_path}/libiomp5md.dll)
      list(APPEND EAGER_OP_IMPL_DEPS ${op_impl_path}/libiomp5md.dll)
    else()
      add_custom_command(
        OUTPUT ${op_impl_path}/openblas.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${OPENBLAS_SHARED_LIB} ${op_impl_path}
        DEPENDS extern_openblas)
      list(APPEND OP_IMPL_DEPS ${op_impl_path}/openblas.dll)
      list(APPEND EAGER_OP_IMPL_DEPS ${op_impl_path}/openblas.dll)
    endif()
    if(WITH_MKLDNN)
      add_custom_command(
        OUTPUT ${op_impl_path}/mkldnn.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_SHARED_LIB} ${op_impl_path}
        DEPENDS mkldnn)
      list(APPEND OP_IMPL_DEPS ${op_impl_path}/mkldnn.dll)
      list(APPEND EAGER_OP_IMPL_DEPS ${op_impl_path}/mkldnn.dll)
    endif()
    if(WITH_ONNXRUNTIME)
      add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/paddle2onnx.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE2ONNX_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS paddle2onnx)
      list(APPEND OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/paddle2onnx.dll)
      list(APPEND EAGER_OP_IMPL_DEPS
           ${CMAKE_CURRENT_BINARY_DIR}/paddle2onnx.dll)

      add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll
        COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_SHARED_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS onnxruntime)
      list(APPEND OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll)
      list(APPEND EAGER_OP_IMPL_DEPS
           ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll)
    endif()

    add_custom_command(
      OUTPUT op_function
      COMMAND
        ${CMAKE_BINARY_DIR}/paddle/fluid/pybind/op_function_generator_retry.bat
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file1}
              ${impl_file1}
      COMMENT "copy_if_different ${tmp_impl_file1} to ${impl_file1}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file2}
              ${impl_file2}
      COMMENT "copy_if_different ${tmp_impl_file2} to ${impl_file2}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file3}
              ${impl_file3}
      COMMENT "copy_if_different ${tmp_impl_file3} to ${impl_file3}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file4}
              ${impl_file4}
      COMMENT "copy_if_different ${tmp_impl_file4} to ${impl_file4}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file5}
              ${impl_file5}
      COMMENT "copy_if_different ${tmp_impl_file5} to ${impl_file5}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file6}
              ${impl_file6}
      COMMENT "copy_if_different ${tmp_impl_file6} to ${impl_file6}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file7}
              ${impl_file7}
      COMMENT "copy_if_different ${tmp_impl_file7} to ${impl_file7}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file8}
              ${impl_file8}
      COMMENT "copy_if_different ${tmp_impl_file8} to ${impl_file8}"
      DEPENDS ${OP_IMPL_DEPS})
    if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
      add_custom_command(
        OUTPUT ${eager_impl_file}
        COMMAND
          ${CMAKE_BINARY_DIR}/paddle/fluid/pybind/eager_legacy_op_function_generator_retry.bat
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_eager_impl_file}
                ${eager_impl_file}
        COMMENT "copy_if_different ${tmp_eager_impl_file} to ${eager_impl_file}"
        DEPENDS ${EAGER_OP_IMPL_DEPS})
    endif()
  else()
    # If there are no *.so in /usr/lib or LD_LIBRARY_PATH,
    # copy these *.so to current directory and append current directory to
    # LD_LIBRARY_PATH. This is different with Windows platformm, which search
    # *.dll in current directory automatically.
    if(WITH_ONNXRUNTIME)
      set(PADDLE2ONNX_PYBIND_OUT
          ${CMAKE_CURRENT_BINARY_DIR}/${PADDLE2ONNX_LIB_NAME})
      set(ONNXRUNTIME_PYBIND_OUT
          ${CMAKE_CURRENT_BINARY_DIR}/${ONNXRUNTIME_LIB_NAME})

      add_custom_command(
        OUTPUT ${PADDLE2ONNX_PYBIND_OUT}
        COMMAND ${CMAKE_COMMAND} -E copy ${PADDLE2ONNX_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS paddle2onnx)
      list(APPEND OP_IMPL_DEPS ${PADDLE2ONNX_PYBIND_OUT})
      list(APPEND EAGER_OP_IMPL_DEPS ${PADDLE2ONNX_PYBIND_OUT})

      add_custom_command(
        OUTPUT ${ONNXRUNTIME_PYBIND_OUT}
        COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS onnxruntime)
      list(APPEND OP_IMPL_DEPS ${ONNXRUNTIME_PYBIND_OUT})
      list(APPEND EAGER_OP_IMPL_DEPS ${ONNXRUNTIME_PYBIND_OUT})
    endif()

    if(WITH_MKLML)
      add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libiomp5.so
        COMMAND ${CMAKE_COMMAND} -E copy ${MKLML_SHARED_IOMP_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS mklml)
      list(APPEND OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/libiomp5.so)
      list(APPEND EAGER_OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/libiomp5.so)
    endif()
    if(WITH_MKLDNN)
      add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libdnnl.so.0
        COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_SHARED_LIB}
                ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS mkldnn)
      list(APPEND OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/libdnnl.so.0)
      list(APPEND EAGER_OP_IMPL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/libdnnl.so.0)
    endif()
    add_custom_command(
      OUTPUT op_function
      COMMAND
        ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:."
        "${CMAKE_CURRENT_BINARY_DIR}/op_function_generator"
        "${op_function_output_path}" "${CODE_GEN_SPLIT_FILE_COUNT}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file1}
              ${impl_file1}
      COMMENT "copy_if_different ${tmp_impl_file1} to ${impl_file1}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file2}
              ${impl_file2}
      COMMENT "copy_if_different ${tmp_impl_file2} to ${impl_file2}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file3}
              ${impl_file3}
      COMMENT "copy_if_different ${tmp_impl_file3} to ${impl_file3}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file4}
              ${impl_file4}
      COMMENT "copy_if_different ${tmp_impl_file4} to ${impl_file4}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file5}
              ${impl_file5}
      COMMENT "copy_if_different ${tmp_impl_file5} to ${impl_file5}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file6}
              ${impl_file6}
      COMMENT "copy_if_different ${tmp_impl_file6} to ${impl_file6}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file7}
              ${impl_file7}
      COMMENT "copy_if_different ${tmp_impl_file7} to ${impl_file7}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_impl_file8}
              ${impl_file8}
      COMMENT "copy_if_different ${tmp_impl_file8} to ${impl_file8}"
      DEPENDS ${OP_IMPL_DEPS}
      VERBATIM)
    if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
      add_custom_command(
        OUTPUT ${eager_impl_file}
        COMMAND
          ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:."
          "${CMAKE_CURRENT_BINARY_DIR}/eager_legacy_op_function_generator"
          "${tmp_eager_impl_file}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${tmp_eager_impl_file}
                ${eager_impl_file}
        COMMENT "copy_if_different ${tmp_eager_impl_file} to ${eager_impl_file}"
        DEPENDS ${EAGER_OP_IMPL_DEPS}
        VERBATIM)
    endif()
  endif()
  add_custom_target(op_function_generator_cmd ALL DEPENDS op_function)
  if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    add_custom_target(eager_legacy_op_function_generator_cmd ALL
                      DEPENDS ${eager_impl_file})
  endif()

  list(APPEND PYBIND_DEPS standalone_executor staticgraph_executor_statistics)

  if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    set(PYBIND_SRCS eager.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_functions.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_method.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_properties.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_utils.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_py_layer.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_legacy_op_function.cc ${PYBIND_SRCS})
    set(PYBIND_SRCS eager_op_function.cc ${PYBIND_SRCS})
    list(APPEND PYBIND_DEPS eager_api)
    list(APPEND PYBIND_DEPS autograd_meta)
    list(APPEND PYBIND_DEPS backward)
    list(APPEND PYBIND_DEPS grad_node_info)
    list(APPEND PYBIND_DEPS phi)
    list(APPEND PYBIND_DEPS final_dygraph_function)
    list(APPEND PYBIND_DEPS final_dygraph_node)
    list(APPEND PYBIND_DEPS dygraph_function)
    list(APPEND PYBIND_DEPS dygraph_node)
    list(APPEND PYBIND_DEPS accumulation_node)
    list(APPEND PYBIND_DEPS py_layer_node)
    list(APPEND PYBIND_DEPS global_utils)
    list(APPEND PYBIND_DEPS utils)
    list(APPEND PYBIND_DEPS python)
    list(APPEND PYBIND_DEPS custom_operator)
    list(APPEND PYBIND_DEPS custom_operator_node)
  endif()

  cc_library(
    paddle_pybind SHARED
    SRCS ${PYBIND_SRCS}
    DEPS ${PYBIND_DEPS} ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})

  if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    add_dependencies(paddle_pybind legacy_eager_codegen)
    add_dependencies(paddle_pybind eager_legacy_op_function_generator_cmd)
  endif()

  if(NOT APPLE AND NOT WIN32)
    target_link_libraries(paddle_pybind rt)
  endif()

  if(WITH_ROCM)
    target_link_libraries(paddle_pybind ${ROCM_HIPRTC_LIB})
  endif()

  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(paddle_pybind ${os_dependency_modules})
  add_dependencies(paddle_pybind op_function_generator_cmd)
endif()
