op_library(lite_engine_op DEPS lite_engine lite_tensor_utils)
<<<<<<< HEAD
cc_test_old(test_lite_engine_op SRCS lite_engine_op_test.cc DEPS lite_engine_op
            analysis)
=======
cc_test(
  test_lite_engine_op
  SRCS lite_engine_op_test.cc
  DEPS lite_engine_op analysis)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
