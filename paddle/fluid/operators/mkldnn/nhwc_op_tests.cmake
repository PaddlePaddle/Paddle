<<<<<<< HEAD
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
=======
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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
