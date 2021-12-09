// CHECK-LABEL: test_tensor_type
func @test_tensor_type() {
  %a = dt.create_uninit_tensor.f32 [3, 4] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%a : !infrt.tensor<X86, NCHW, F32>) {value=1.0:f32}
  // CHECK: tensor: shape=shape[3,4], values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  dt.print_tensor (%a : !infrt.tensor<X86, NCHW, F32>)

  infrt.return
}
