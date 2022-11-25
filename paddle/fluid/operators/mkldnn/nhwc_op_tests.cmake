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
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
