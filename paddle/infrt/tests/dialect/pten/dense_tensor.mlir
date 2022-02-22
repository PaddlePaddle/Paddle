// RUN: infrtexec -i %s | FileCheck %s

// CHECK-LABEL: @fake_pten_kernel_execute
func @fake_pten_kernel_execute() {
  %allocator = "pten_dt.create_allocator.cpu" (): () -> !pten.CPU_allocator
  %ctx = "pten_dt.create_context.cpu" (): () -> !pten.CPU_context
  %t = "pten_dt.create_dense_tensor.cpu.f32.nchw" (%allocator) {dims=[1:i64], lod=[1:i64]}: (!pten.CPU_allocator) -> (!infrt.tensor<X86, NCHW, F32>)

  // %d = "pten_dt.fake_pten_kernel" (%ctx, %t, %t) {transpose_x=false, transpose_y=false} : (!pten.CPU_context, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> (!infrt.tensor<X86, NCHW, F32>)
  infrt.return
}
