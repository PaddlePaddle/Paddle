cc_test(
<<<<<<< HEAD
=======
  test_egr_task_tensor_utils
  SRCS tensor_utils_test.cc
  DEPS ${eager_deps})
cc_test(
  test_egr_task_eager_utils
  SRCS eager_utils_test.cc
  DEPS ${eager_deps})
cc_test(
  test_egr_task_forward_autograd
  SRCS forward_autograd_test.cc
  DEPS ${eager_deps} ${fluid_deps} eager_scale scale_node)
cc_test(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  test_egr_task_nan_inf_utils
  SRCS nan_inf_utils_test.cc
  DEPS eager_nan_inf_utils)

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  cc_test(
    test_egr_task_hook
    SRCS hook_test.cc
    DEPS ${eager_deps} ${fluid_deps} ${generated_deps} eager_scale scale_node)
  cc_test(
    test_egr_task_backward
    SRCS backward_test.cc
    DEPS ${eager_deps} ${fluid_deps} ${generated_deps} eager_scale scale_node)
  cc_test(
    test_egr_task_grad
    SRCS grad_test.cc
    DEPS ${eager_deps} ${fluid_deps} ${generated_deps} eager_scale scale_node)
  cc_test(
    test_egr_task_fwd_bwd_joint
    SRCS fwd_bwd_joint_test.cc
    DEPS ${eager_deps} ${fluid_deps} ${generated_deps} eager_scale scale_node)
  cc_test(
    test_egr_task_cross_batch
    SRCS cross_batch_accumulation_test.cc
    DEPS ${eager_deps} ${fluid_deps} ${generated_deps} eager_scale scale_node)
  cc_test(
    test_egr_task_hook_intermidiate
    SRCS hook_test_intermidiate.cc
    DEPS ${eager_deps} ${fluid_deps} ${generated_deps} dygraph_node)
  cc_test(
    test_egr_task_autocodegen
    SRCS generated_test.cc
    DEPS ${eager_deps} ${fluid_deps} ${generated_deps})
<<<<<<< HEAD
  cc_test(
    test_egr_task_tensor_utils
    SRCS tensor_utils_test.cc
    DEPS ${eager_deps} ${generated_deps})
  cc_test(
    test_egr_task_eager_utils
    SRCS eager_utils_test.cc
    DEPS ${eager_deps} ${generated_deps})
  cc_test(
    test_egr_task_forward_autograd
    SRCS forward_autograd_test.cc
    DEPS ${eager_deps} ${fluid_deps} ${generated_deps} eager_scale scale_node)
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
endif()
