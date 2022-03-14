// RUN: infrtexec -i %s | FileCheck %s
// CHECK-LABEL: test_tensor_type
func @test_tensor_type() {
  %a = dt.create_uninit_tensor.f32 [3, 4] -> !infrt.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%a : !infrt.dense_tensor<CPU, FP32, NCHW>) {value=1.0:f32}
  // CHECK: tensor: shape=shape[3,4], values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  dt.print_tensor (%a : !infrt.dense_tensor<CPU, FP32, NCHW>)

  infrt.return
}
