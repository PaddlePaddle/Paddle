// RUN: infrtopt %s | FileCheck %s

// CHECK-LABEL: @basic_tensor
func @basic_tensor() {
  %a = "pten_dt.create_allocator.cpu" (): () -> !pten.CPU_allocator
  %b = "pten_dt.create_context.cpu" (): () -> !pten.CPU_context
  %c = "pten_dt.create_dense_tensor.cpu.f32.nchw" (%a) {dims=[1:i64], lod=[1:i64]}: (!pten.CPU_allocator) -> (!infrt.tensor<X86, NCHW, F32>)
  // "pten_dt.fill_dense_tensor.f32" (%c) {value=[1.0:f32]} : (!infrt.tensor<X86, NCHW, F32>) -> ()
  %d = "pten_kernel.matmul.host.fp32" (%a, %c, %c) {transpose_x=false, transpose_y=false} : (!pten.CPU_allocator, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> (!infrt.tensor<X86, NCHW, F32>)

  infrt.return
}
