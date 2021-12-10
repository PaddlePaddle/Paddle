// CHECK-LABEL: @predict
func @predict(%input:!infrt.tensor<X86, NCHW, F32>, %map: !infrt.tensor_map) -> (!infrt.tensor<X86, NCHW, F32>) {
  %w = dt.get_param(%map, "create_parameter_0.w_0") -> !infrt.tensor<X86, NCHW, F32>
  %bias = dt.get_param(%map, "create_parameter_1.w_0") -> !infrt.tensor<X86, NCHW, F32>

  %out = dt.create_uninit_tensor.f32 [3, 3] -> !infrt.tensor<X86, NCHW, F32>

  // fc
  "external.matmul"(%input, %w, %out) {}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  //dt.print_tensor (%out : !infrt.tensor<X86, NCHW, F32>)

  infrt.return %out : !infrt.tensor<X86, NCHW, F32>
}

// CHECK-LABEL: @main
func @main() {
  %input = dt.create_uninit_tensor.f32 [3, 3] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%input : !infrt.tensor<X86, NCHW, F32>) {value=1.0:f32}

  %path = infrt.get_string("/infrt/build/paddle/paddle_1.8_fc_model")
  // CHECK-LABEL: loading params
  %map = dt.load_params(%path)

  %out = infrt.call @predict(%input, %map): (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor_map) -> (!infrt.tensor<X86, NCHW, F32>)
  dt.print_tensor (%out : !infrt.tensor<X86, NCHW, F32>)

  infrt.return
}

