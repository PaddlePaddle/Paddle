// RUN: infrtexec -i %s | FileCheck %s
// CHECK-LABEL: dense_shape0
func @dense_shape0() {
  %shape = ts.build_shape [1:i64, 57:i64]
  %a = dt.create_uninit_tensor.f32 [12:i64, 23:i64] -> !infrt.dense_tensor<CPU, FP32, NCHW>

  infrt.return
}

func @predict(%a: !infrt.dense_tensor<CPU, FP32, NCHW>, %b: !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) {
  %a0 = dt.shallow_copy_tensor %a : !infrt.dense_tensor<CPU, FP32, NCHW> -> !infrt.dense_tensor<CPU, FP32, NCHW>
  %b0 = dt.shallow_copy_tensor %b : !infrt.dense_tensor<CPU, FP32, NCHW> -> !infrt.dense_tensor<CPU, FP32, NCHW>

  infrt.return %a0, %b0: !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>
}


func @main() {
  %shape = ts.build_shape [1:i64, 57:i64]
  %a = dt.create_uninit_tensor.f32 [12:i64, 23:i64] -> !infrt.dense_tensor<CPU, FP32, NCHW>

  %b, %c = infrt.call @predict(%a, %a) : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
  infrt.return
}
