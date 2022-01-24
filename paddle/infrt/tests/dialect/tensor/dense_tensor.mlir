// RUN: infrtexec -i %s | FileCheck %s
// CHECK-LABEL: dense_shape0
func @dense_shape0() {
  %shape = ts.build_shape [1:i64, 57:i64]
  %a = dt.create_uninit_tensor.f32 [12:i64, 23:i64] -> !infrt.tensor<X86, NCHW, F32>

  infrt.return
}

func @predict(%a: !infrt.tensor<X86, NCHW, F32>, %b: !infrt.tensor<X86, NCHW, F32>) -> (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) {
  %a0 = dt.shallow_copy_tensor %a : !infrt.tensor<X86, NCHW, F32> -> !infrt.tensor<X86, NCHW, F32>
  %b0 = dt.shallow_copy_tensor %b : !infrt.tensor<X86, NCHW, F32> -> !infrt.tensor<X86, NCHW, F32>

  infrt.return %a0, %b0: !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>
}


func @main() {
  %shape = ts.build_shape [1:i64, 57:i64]
  %a = dt.create_uninit_tensor.f32 [12:i64, 23:i64] -> !infrt.tensor<X86, NCHW, F32>

  %b, %c = infrt.call @predict(%a, %a) : (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>)
  infrt.return
}
