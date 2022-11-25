op_library(lite_engine_op DEPS lite_engine lite_tensor_utils)
<<<<<<< HEAD
cc_test_old(test_lite_engine_op SRCS lite_engine_op_test.cc DEPS lite_engine_op
            analysis)
=======
cc_test(
  test_lite_engine_op
  SRCS lite_engine_op_test.cc
  DEPS lite_engine_op analysis)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
