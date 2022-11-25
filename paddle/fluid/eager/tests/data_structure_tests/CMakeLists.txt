<<<<<<< HEAD
cc_test_old(test_egr_ds_eager_tensor SRCS eager_tensor_test.cc DEPS
            ${eager_deps})
cc_test_old(test_egr_ds_auotgrad_meta SRCS autograd_meta_test.cc DEPS
            ${eager_deps})
cc_test_old(test_egr_ds_grad_node_info SRCS grad_node_info_test.cc DEPS
            ${eager_deps})
cc_test_old(test_egr_ds_accumulation_node SRCS accumulation_node_test.cc DEPS
            ${eager_deps})
cc_test_old(test_egr_ds_tensor_wrapper SRCS tensor_wrapper_test.cc DEPS
            ${eager_deps})

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  cc_test_old(test_egr_ds_grad_tensor_holder SRCS grad_tensor_holder_test.cc
              DEPS ${eager_deps} ${generated_deps})
=======
cc_test(
  test_egr_ds_eager_tensor
  SRCS eager_tensor_test.cc
  DEPS ${eager_deps})
cc_test(
  test_egr_ds_auotgrad_meta
  SRCS autograd_meta_test.cc
  DEPS ${eager_deps})
cc_test(
  test_egr_ds_grad_node_info
  SRCS grad_node_info_test.cc
  DEPS ${eager_deps})
cc_test(
  test_egr_ds_accumulation_node
  SRCS accumulation_node_test.cc
  DEPS ${eager_deps})
cc_test(
  test_egr_ds_tensor_wrapper
  SRCS tensor_wrapper_test.cc
  DEPS ${eager_deps})

if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
  cc_test(
    test_egr_ds_grad_tensor_holder
    SRCS grad_tensor_holder_test.cc
    DEPS ${eager_deps} ${generated_deps})
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
endif()
