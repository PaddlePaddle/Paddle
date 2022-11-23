op_library(lite_engine_op DEPS lite_engine lite_tensor_utils)
<<<<<<< HEAD
cc_test(
  test_lite_engine_op
  SRCS lite_engine_op_test.cc
  DEPS lite_engine_op analysis)
=======
cc_test_old(test_lite_engine_op SRCS lite_engine_op_test.cc DEPS lite_engine_op
            analysis)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
