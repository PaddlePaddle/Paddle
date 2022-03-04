// CHECK-LABEL: @predict
func @predict(%input:!Infrt.tensor<X86, NCHW, F32>, %map: !Infrt.tensor_map) -> (!Infrt.tensor<X86, NCHW, F32>) {
  %w = dt.get_param(%map, "create_parameter_0.w_0") -> !Infrt.tensor<X86, NCHW, F32>
  %bias = dt.get_param(%map, "create_parameter_1.w_0") -> !Infrt.tensor<X86, NCHW, F32>

  %out = dt.create_uninit_tensor.f32 [3, 3] -> !Infrt.tensor<X86, NCHW, F32>

  // fc
  "external.matmul"(%input, %w, %out) {}: (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> ()
  //dt.print_tensor (%out : !Infrt.tensor<X86, NCHW, F32>)

  Infrt.return %out : !Infrt.tensor<X86, NCHW, F32>
}

// CHECK-LABEL: @main
func @main() {
  %input = dt.create_uninit_tensor.f32 [3, 3] -> !Infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%input : !Infrt.tensor<X86, NCHW, F32>) {value=1.0:f32}

  %path = Infrt.get_string("/Infrt/build/paddle/paddle_1.8_fc_model")
  // CHECK-LABEL: loading params
  %map = dt.load_params(%path)

  %out = Infrt.call @predict(%input, %map): (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor_map) -> (!Infrt.tensor<X86, NCHW, F32>)
  dt.print_tensor (%out : !Infrt.tensor<X86, NCHW, F32>)

  Infrt.return
}

