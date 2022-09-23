// CHECK-LABEL: @predict
func @predict(%input:!infrt.dense_tensor<CPU, FP32, NCHW>, %map: !infrt.dense_tensor_map) -> (!infrt.dense_tensor<CPU, FP32, NCHW>) {
  %w = dt.get_param(%map, "create_parameter_0.w_0") -> !infrt.dense_tensor<CPU, FP32, NCHW>
  %bias = dt.get_param(%map, "create_parameter_1.w_0") -> !infrt.dense_tensor<CPU, FP32, NCHW>

  %out = dt.create_uninit_tensor.f32 [3, 3] -> !infrt.dense_tensor<CPU, FP32, NCHW>

  // fc
  "external.matmul"(%input, %w, %out) {}: (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
  "external.sigmoid"(%out, %out) {}: (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
  //dt.print_tensor (%out : !infrt.dense_tensor<CPU, FP32, NCHW>)

  infrt.return %out : !infrt.dense_tensor<CPU, FP32, NCHW>
}

// CHECK-LABEL: @main
func @main() {
  %input = dt.create_uninit_tensor.f32 [3, 3] -> !infrt.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%input : !infrt.dense_tensor<CPU, FP32, NCHW>) {value=1.0:f32}

  // CHECK-LABEL: loading params
  %map = dt.load_params() {path="/Infrt/build/paddle/paddle_1.8_fc_model"}

  %out = infrt.call @predict(%input, %map): (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor_map) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
  dt.print_tensor (%out : !infrt.dense_tensor<CPU, FP32, NCHW>)

  infrt.return
}

