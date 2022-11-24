op_library(lite_engine_op DEPS lite_engine lite_tensor_utils)
<<<<<<< HEAD
cc_test(
  test_lite_engine_op
  SRCS lite_engine_op_test.cc
  DEPS lite_engine_op analysis)
=======
cc_test_old(test_lite_engine_op SRCS lite_engine_op_test.cc DEPS lite_engine_op
            analysis)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
