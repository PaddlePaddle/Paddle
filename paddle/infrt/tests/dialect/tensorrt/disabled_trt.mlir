// RUN: infrtexec -i %s | FileCheck %s

// CHECK-LABEL: @run_trt
func @run_trt(%0 : !infrt.dense_tensor<GPU, FP32, NCHW>, %ctx : !phi.context<GPU>) {
  %a = "trt.create_engine"(%0) ({
    %1 = "trt.Activation"(%0) {activation_type = 1 : si32, alpha = 1.0 : f32, beta = 6.0 : f32} : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    "infrt.return"(%1) : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> ()
  }) : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> !trt.engine
  "trt.inspect_engine"(%a) {} : (!trt.engine) -> ()

  %res = "trt.compute"(%a, %ctx) {} : (!trt.engine, !phi.context<GPU>) -> (!infrt.tensor_list)
  %size = "dt.tensor_list_get_size"(%res) {} : (!infrt.tensor_list) -> (i32)
  "infrt.print.i32"(%size) {} : (i32) -> ()

  %ts0 = "dt.tensor_list_get_tensor"(%res) {id = 0 : i32} : (!infrt.tensor_list) -> (!infrt.dense_tensor<GPU, FP32, NCHW>)
  "phi_dt.print_tensor" (%ts0) : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> ()

  infrt.return
}

// CHECK-LABEL: @main
func @main() {
  %ctx = "phi_dt.create_context.gpu" (): () -> !phi.context<GPU>
  %t = "phi_dt.create_dense_tensor.gpu" (%ctx) {
    precision=#infrt.precision<FP32>,
    layout=#infrt.layout<NCHW>,
    dims=[1:i64, 3:i64, 1:i64, 1:i64], lod=[1:i64]}: (!phi.context<GPU>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>)

  "phi_dt.fill_dense_tensor.f32"(%t) {value=[3.8:f32, 2.4:f32, 1.3:f32]} : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> ()
  "phi_dt.print_tensor" (%t) : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> ()

  //%res = 
  infrt.call @run_trt(%t, %ctx) : (!infrt.dense_tensor<GPU, FP32, NCHW>, !phi.context<GPU>) -> ()
  //-> (!infrt.dense_tensor<GPU, FP32, NCHW>)

  infrt.return
}
