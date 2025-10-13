# OneDNN IR Pass Tests

cc_test(
  test_depthwise_conv_onednn_pass
  SRCS depthwise_conv_onednn_pass_test.cc
  DEPS depthwise_conv_onednn_pass)

cc_test(
  test_int8_scale_calculation_onednn_pass
  SRCS int8_scale_calculation_onednn_pass_test.cc
  DEPS int8_scale_calculation_onednn_pass pass_test_util)

cc_test(
  test_params_quantization_onednn_pass
  SRCS params_quantization_onednn_pass_test.cc
  DEPS params_quantization_onednn_pass)

cc_test(
  test_onednn_placement_pass
  SRCS onednn_placement_pass_test.cc
  DEPS onednn_placement_pass)

cc_test(
  test_compute_propagate_scales_onednn_pass
  SRCS compute_propagate_scales_onednn_pass_test.cc
  DEPS compute_propagate_scales_onednn_pass naive_executor)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(test_compute_propagate_scales_onednn_pass)
endif()

cc_test(
  test_cpu_quantize_placement_pass
  SRCS cpu_quantize_placement_pass_test.cc
  DEPS cpu_quantize_placement_pass)

cc_test(
  test_cpu_quantize_pass
  SRCS cpu_quantize_pass_test.cc
  DEPS cpu_quantize_pass naive_executor)

cc_test(
  test_cpu_quantize_squash_pass
  SRCS cpu_quantize_squash_pass_test.cc
  DEPS cpu_quantize_squash_pass naive_executor)

cc_test(
  test_shuffle_channel_onednn_detect_pass
  SRCS shuffle_channel_onednn_detect_pass_test.cc
  DEPS shuffle_channel_onednn_detect_pass)

cc_test(
  test_cpu_bfloat16_placement_pass
  SRCS cpu_bfloat16_placement_pass_test.cc
  DEPS cpu_bfloat16_placement_pass)

cc_test(
  test_cpu_bfloat16_pass
  SRCS cpu_bfloat16_pass_test.cc
  DEPS cpu_bfloat16_pass)
