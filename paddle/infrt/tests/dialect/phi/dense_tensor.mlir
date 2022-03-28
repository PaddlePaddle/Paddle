// RUN: infrtexec -i %s | FileCheck %s

// CHECK-LABEL: @sign_any_float32_execute
func @sign_any_float32_execute() {
  %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
  %t = "phi_dt.create_dense_tensor.cpu" (%ctx) {
    precision=#core.precision<FP32>, 
    layout=#core.layout<NCHW>, lod=[1:i64], dims=[1:i64]}: (!phi.context<CPU>) -> (!core.dense_tensor<CPU, FP32, NCHW>)
  "phi_dt.fill_dense_tensor.f32"(%t) {value=[3.8:f32]} : (!core.dense_tensor<CPU, FP32, NCHW>) -> ()
  %e = "phi_cpu.sign.float32.any"(%ctx, %t) : (!phi.context<CPU>, !core.dense_tensor<CPU, FP32, NCHW>) -> (!core.dense_tensor<CPU, FP32, NCHW>)

  // CHECK: dense_tensor: shape=shape[1], value=[1]
  "phi_dt.print_tensor" (%e) : (!core.dense_tensor<CPU, FP32, NCHW>) -> ()
  core.return
}

