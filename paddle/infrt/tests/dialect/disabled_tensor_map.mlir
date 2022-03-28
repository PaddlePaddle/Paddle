// CHECK-LABEL: @predict
func @predict(%input:!core.dense_tensor<CPU, FP32, NCHW>, %map: !core.dense_tensor_map) -> (!core.dense_tensor<CPU, FP32, NCHW>) {
  %w = dt.get_param(%map, "create_parameter_0.w_0") -> !core.dense_tensor<CPU, FP32, NCHW>
  %bias = dt.get_param(%map, "create_parameter_1.w_0") -> !core.dense_tensor<CPU, FP32, NCHW>

  %out = dt.create_uninit_tensor.f32 [3, 3] -> !core.dense_tensor<CPU, FP32, NCHW>

  // fc
  "external.matmul"(%input, %w, %out) {}: (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> ()
  "external.sigmoid"(%out, %out) {}: (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> ()
  //dt.print_tensor (%out : !core.dense_tensor<CPU, FP32, NCHW>)

  core.return %out : !core.dense_tensor<CPU, FP32, NCHW>
}

// CHECK-LABEL: @main
func @main() {
  %input = dt.create_uninit_tensor.f32 [3, 3] -> !core.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%input : !core.dense_tensor<CPU, FP32, NCHW>) {value=1.0:f32}

  // CHECK-LABEL: loading params
  %map = dt.load_params() {path="/Infrt/build/paddle/paddle_1.8_fc_model"}

  %out = core.call @predict(%input, %map): (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor_map) -> (!core.dense_tensor<CPU, FP32, NCHW>)
  dt.print_tensor (%out : !core.dense_tensor<CPU, FP32, NCHW>)

  core.return
}

