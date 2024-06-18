set(prim_eager_deps
    phi
    common
    hook_utils
    utils
    global_utils
    backward
    tracer
    layer
    autograd_meta
    eager_nan_inf_utils
    grad_node_info
    grad_tensor_holder
    custom_operator_node)

set(prim_generated_deps final_dygraph_function final_dygraph_node
                        dygraph_function dygraph_node)

if(NOT (NOT WITH_PYTHON AND ON_INFER))
  if(WITH_CINN)
    set(prim_eager_deps ${prim_eager_deps})
  endif()
  cc_library(init_env_utils SRCS init_env_utils.cc)
  target_compile_definitions(init_env_utils PUBLIC PADDLE_DLL_EXPORT)

  paddle_test(test_comp_eager SRCS test_eager_prim.cc init_env_utils.cc)
endif()

# skip win32 since wget is not installed by default on windows machine.

if(NOT WIN32)
  paddle_test(test_vjp_pir SRCS test_vjp.cc)
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_comp_eager)
endif()
