func @dense_shape0() {
  %shape = ts.build_shape [1:i64, 57:i64]
  %a = dt.create_uninit_tensor.f32 [12:i64, 23:i64] -> !cinn.tensor<X86, NCHW, F32>

  cinn.return
}

func @predict(%a: !cinn.tensor<X86, NCHW, F32>, %b: !cinn.tensor<X86, NCHW, F32>) -> (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) {
  %a0 = dt.shallow_copy_tensor %a : !cinn.tensor<X86, NCHW, F32> -> !cinn.tensor<X86, NCHW, F32>
  %b0 = dt.shallow_copy_tensor %b : !cinn.tensor<X86, NCHW, F32> -> !cinn.tensor<X86, NCHW, F32>

  cinn.return %a0, %b0: !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>
}


func @main() {
  %shape = ts.build_shape [1:i64, 57:i64]
  %a = dt.create_uninit_tensor.f32 [12:i64, 23:i64] -> !cinn.tensor<X86, NCHW, F32>

  %b, %c = cinn.call @predict(%a, %a) : (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>)
  cinn.return
}
