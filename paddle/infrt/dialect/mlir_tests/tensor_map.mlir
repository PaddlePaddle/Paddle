// CHECK-LABEL: @predict
func @predict(%input:!cinn.tensor<X86, NCHW, F32>, %map: !cinn.tensor_map) -> (!cinn.tensor<X86, NCHW, F32>) {
  %w = dt.get_param(%map, "create_parameter_0.w_0") -> !cinn.tensor<X86, NCHW, F32>
  %bias = dt.get_param(%map, "create_parameter_1.w_0") -> !cinn.tensor<X86, NCHW, F32>

  %out = dt.create_uninit_tensor.f32 [3, 3] -> !cinn.tensor<X86, NCHW, F32>

  // fc
  "external.matmul"(%input, %w, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  //dt.print_tensor (%out : !cinn.tensor<X86, NCHW, F32>)

  cinn.return %out : !cinn.tensor<X86, NCHW, F32>
}

// CHECK-LABEL: @main
func @main() {
  %input = dt.create_uninit_tensor.f32 [3, 3] -> !cinn.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%input : !cinn.tensor<X86, NCHW, F32>) {value=1.0:f32}

  %path = cinn.get_string("/cinn/build/paddle/paddle_1.8_fc_model")
  // CHECK-LABEL: loading params
  %map = dt.load_params(%path)

  %out = cinn.call @predict(%input, %map): (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor_map) -> (!cinn.tensor<X86, NCHW, F32>)
  dt.print_tensor (%out : !cinn.tensor<X86, NCHW, F32>)

  cinn.return
}

