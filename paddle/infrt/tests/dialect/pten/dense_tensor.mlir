// RUN: infrtopt %s | FileCheck %s

// CHECK-LABEL: basic_tensor
func @basic_tensor() {
  %a = "pten_dt.create_uninit_tensor.f32" () { shape= [12:i64, 23:i64] } : () -> !infrt.tensor<X86, NCHW, F32>
}
