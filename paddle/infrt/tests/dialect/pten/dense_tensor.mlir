// RUN: infrtexec -i %s | FileCheck %s

// CHECK-LABEL: @sign_any_float32_execute
func @sign_any_float32_execute() {
  %allocator = "phi_dt.create_allocator.cpu" (): () -> !phi.CPU_allocator
  %ctx = "phi_dt.create_context.cpu" (%allocator): (!phi.CPU_allocator) -> !phi.CPU_context
  %t = "phi_dt.create_dense_tensor.cpu.f32.nchw" (%allocator) {dims=[1:i64], lod=[1:i64]}: (!phi.CPU_allocator) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
  "phi_dt.fill_dense_tensor.f32"(%t) {value=[3.8:f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
  %e = "phi_cpu.sign.any.float32"(%ctx, %t) : (!phi.CPU_context, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)

  // CHECK: dense_tensor: shape=shape[1], values=[1]
  "phi_dt.print_tensor" (%e) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
  Infrt.return
}

