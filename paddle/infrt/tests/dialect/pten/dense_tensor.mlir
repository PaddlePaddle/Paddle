// RUN: infrtopt %s | FileCheck %s

// CHECK-LABEL: @basic_tensor
func @basic_tensor() {
  %a = "phi_dt.create_allocator.cpu" (): () -> !phi.CPU_allocator
  %b = "phi_dt.create_context.cpu" (): () -> !phi.CPU_context
  %c = "phi_dt.create_dense_tensor.cpu.f32.nchw" (%a) {dims=[1:i64], lod=[1:i64]}: (!phi.CPU_allocator) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
  // "phi_dt.fill_dense_tensor.f32" (%c) {value=[1.0:f32]} : (!Infrt.tensor<CPU, FP32, NCHW>) -> ()

  Infrt.return
}
