// RUN: infrtopt %s | FileCheck %s

// CHECK-LABEL: basic_tensor
func @basic_tensor() {
  %a = "pten_dt.create_uninit_tensor.f32" () { shape=[12:i64, 23:i64] } : () -> !infrt.tensor<X86, NCHW, F32>
  %b = "pten_dt.create_inited_tensor.f32" () { shape=[2:i64, 2:i64], values=[0.1:f32, 0.2:f32, 0.3:f32, 0.4:f32] } : () -> !infrt.tensor<X86, NCHW, F32>
  "pten_dt.fill_tensor_with_constant.f32" (%a) { value=0.1:f32 } : (!infrt.tensor<X86, NCHW, F32>) -> ()

  infrt.return
}
