// RUN: infrtexec -i %s | FileCheck %s

// CHECK-LABEL: @fake_phi_kernel_execute
func @fake_phi_kernel_execute() {
  %allocator = "phi_dt.create_allocator.cpu" (): () -> !phi.CPU_allocator
  %ctx = "phi_dt.create_context.cpu" (): () -> !phi.CPU_context
  %t = "phi_dt.create_dense_tensor.cpu.f32.nchw" (%allocator) {dims=[1:i64], lod=[1:i64]}: (!phi.CPU_allocator) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)

  // CHECK: @FakePhiKernel@
  %d = "phi_dt.fake_phi_kernel" (%ctx, %t, %t) {transpose_x=false, transpose_y=false} : (!phi.CPU_context, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
  Infrt.return
}

