// RUN: infrtexec -i %s | FileCheck %s
// CHECK-LABEL: naive_elementwise_add
func @naive_elementwise_add() {
  // create a
  %a = dt.create_uninit_tensor.f32 [2:i64, 8:i64] -> !infrt.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%a : !infrt.dense_tensor<CPU, FP32, NCHW>) {value=1.0:f32}
  // create b
  %b = dt.create_uninit_tensor.f32 [2:i64, 8:i64] -> !infrt.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%b : !infrt.dense_tensor<CPU, FP32, NCHW>) {value=2.0:f32}
  // get c
  %c = dt.naive_elementwise_add.f32(%a, %b) {} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>

  // CHECK: tensor: shape=shape[2,8], values=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  dt.print_tensor (%c : !infrt.dense_tensor<CPU, FP32, NCHW>)

  infrt.return
}

// RUN: infrtexec -i %s | FileCheck %s
// CHECK-LABEL: naive_matmul
func @naive_matmul() {
  // create a
  %a = dt.create_uninit_tensor.f32 [2:i64, 8:i64] -> !infrt.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%a : !infrt.dense_tensor<CPU, FP32, NCHW>) {value=1.0:f32}
  // create b
  %b = dt.create_uninit_tensor.f32 [8:i64, 4:i64] -> !infrt.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%b : !infrt.dense_tensor<CPU, FP32, NCHW>) {value=2.0:f32}
  // get c
  %c = dt.naive_matmul.f32(%a, %b) {} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>

  // CHECK: tensor: shape=shape[2,4], values=[16, 16, 16, 16, 16, 16, 16, 16]
  dt.print_tensor (%c : !infrt.dense_tensor<CPU, FP32, NCHW>)

  infrt.return
}
