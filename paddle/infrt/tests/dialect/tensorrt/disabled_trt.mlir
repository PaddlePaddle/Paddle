// RUN: infrtexec -i %s | FileCheck %s

// CHECK-LABEL: @run_trt
func @run_trt(%0 : !core.dense_tensor<GPU, FP32, NCHW>, %ctx : !phi.context<GPU>) {
  %a = "trt.create_engine"(%0) ({
    %1 = "trt.Activation"(%0) {activation_type = 1 : si32, alpha = 1.0 : f32, beta = 6.0 : f32} : (!core.dense_tensor<GPU, FP32, NCHW>) -> !core.dense_tensor<GPU, FP32, NCHW>
    "core.return"(%1) : (!core.dense_tensor<GPU, FP32, NCHW>) -> ()
  }) : (!core.dense_tensor<GPU, FP32, NCHW>) -> !trt.engine
  "trt.inspect_engine"(%a) {} : (!trt.engine) -> ()

  %res = "trt.compute"(%a, %ctx) {} : (!trt.engine, !phi.context<GPU>) -> (!core.tensor_list)
  %size = "dt.tensor_list_get_size"(%res) {} : (!core.tensor_list) -> (i32)
  "core.print.i32"(%size) {} : (i32) -> ()

  %ts0 = "dt.tensor_list_get_tensor"(%res) {id = 0 : i32} : (!core.tensor_list) -> (!core.dense_tensor<GPU, FP32, NCHW>)
  "phi_dt.print_tensor" (%ts0) : (!core.dense_tensor<GPU, FP32, NCHW>) -> ()

  core.return
}

// CHECK-LABEL: @main
func @main() {
  %ctx = "phi_dt.create_context.gpu" (): () -> !phi.context<GPU>
  %t = "phi_dt.create_dense_tensor.gpu" (%ctx) {
    precision=#core.precision<FP32>,
    layout=#core.layout<NCHW>,
    dims=[1:i64, 3:i64, 1:i64, 1:i64], lod=[1:i64]}: (!phi.context<GPU>) -> (!core.dense_tensor<GPU, FP32, NCHW>)

  "phi_dt.fill_dense_tensor.f32"(%t) {value=[3.8:f32, 2.4:f32, 1.3:f32]} : (!core.dense_tensor<GPU, FP32, NCHW>) -> ()
  "phi_dt.print_tensor" (%t) : (!core.dense_tensor<GPU, FP32, NCHW>) -> ()

  //%res = 
  core.call @run_trt(%t, %ctx) : (!core.dense_tensor<GPU, FP32, NCHW>, !phi.context<GPU>) -> ()
  //-> (!core.dense_tensor<GPU, FP32, NCHW>)

  core.return
}
