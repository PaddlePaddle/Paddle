// RUN: infrtexec -i %s | FileCheck %s
// CHECK-LABEL: dense_shape0
func @dense_shape0() {
  %shape = ts.build_shape [1:i64, 57:i64]
  %a = dt.create_uninit_tensor.f32 [12:i64, 23:i64] -> !core.dense_tensor<CPU, FP32, NCHW>

  core.return
}

func @predict(%a: !core.dense_tensor<CPU, FP32, NCHW>, %b: !core.dense_tensor<CPU, FP32, NCHW>) -> (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) {
  %a0 = dt.shallow_copy_tensor %a : !core.dense_tensor<CPU, FP32, NCHW> -> !core.dense_tensor<CPU, FP32, NCHW>
  %b0 = dt.shallow_copy_tensor %b : !core.dense_tensor<CPU, FP32, NCHW> -> !core.dense_tensor<CPU, FP32, NCHW>

  core.return %a0, %b0: !core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>
}


func @main() {
  %shape = ts.build_shape [1:i64, 57:i64]
  %a = dt.create_uninit_tensor.f32 [12:i64, 23:i64] -> !core.dense_tensor<CPU, FP32, NCHW>

  %b, %c = core.call @predict(%a, %a) : (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>)
  core.return
}
