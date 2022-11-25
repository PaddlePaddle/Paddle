<<<<<<< HEAD
cc_test_old(
  test_mkldnn_op_nhwc
  SRCS
  mkldnn/test_mkldnn_op_nhwc.cc
  DEPS
  op_registry
  pool_op
  shape_op
  crop_op
  activation_op
  generated_op
  pooling
  transpose_op
  scope
  device_context
  enforce
  executor)
=======
cc_test(
  test_mkldnn_op_nhwc
  SRCS mkldnn/test_mkldnn_op_nhwc.cc
  DEPS op_registry
       pool_op
       shape_op
       crop_op
       activation_op
       pooling
       transpose_op
       scope
       device_context
       enforce
       executor)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
