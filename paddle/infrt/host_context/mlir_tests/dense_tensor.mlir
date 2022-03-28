// CHECK-LABEL: build_tensor1
func @build_tensor1() {
  %a = dt.create_uninit_tensor.f32 [3, 4] -> !core.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%a : !core.dense_tensor<CPU, FP32, NCHW>) {value=1.0:f32}
  // CHECK: tensor: shape=shape[3,4], values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  dt.print_tensor (%a : !core.dense_tensor<CPU, FP32, NCHW>)

  core.return
}
