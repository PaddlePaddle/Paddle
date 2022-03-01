// RUN: infrtexec -i %s | FileCheck %s

// CHECK-LABEL: @cpu.matmul_execute
func @cpu.matmul_execute() {
  %allocator = "phi_dt.create_allocator.cpu" (): () -> !phi.allocator<CPU>
  %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
  %t = "phi_dt.create_dense_tensor.cpu.f32.nchw" (%allocator) {dims=[1:i64], lod=[1:i64]}:
    (!phi.allocator<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
  %d = "phi_kernel.cpu.matmul" (%ctx, %t, %t) {transpose_x=false, transpose_y=false} :
    (!phi.context<CPU>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
  Infrt.return
}

