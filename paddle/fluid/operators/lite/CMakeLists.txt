op_library(lite_engine_op DEPS lite_engine lite_tensor_utils)
<<<<<<< HEAD
cc_test_old(test_lite_engine_op SRCS lite_engine_op_test.cc DEPS lite_engine_op
            analysis)
=======
cc_test(
  test_lite_engine_op
  SRCS lite_engine_op_test.cc
  DEPS lite_engine_op analysis)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
