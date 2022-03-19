// RUN: infrtexec -i %s | FileCheck %s

// CHECK-LABEL: @run_trt
func @run_trt(%input_tensor : !infrt.dense_tensor<GPU, FP32, NCHW>, %kernel_weight : !infrt.dense_tensor<CPU, FP32, NCHW>, %kernel_bias : !infrt.dense_tensor<CPU, FP32, NCHW>, %gpu_ctx : !phi.context<GPU>) {
  %a = "trt.create_engine"(%input_tensor, %kernel_weight, %kernel_bias) ({
    %1 = "trt.Activation"(%input_tensor) {activation_type = 1 : si32, alpha = 1.0 : f32, beta = 6.0 : f32} : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %2 = "trt.Convolution"(%input_tensor, %kernel_weight, %kernel_bias) {out_channel_num = 3 : si32, kernel_size = [3:i32, 3:i32]} : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    "infrt.return"(%1, %2) : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> ()
  }) : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !trt.engine
  "trt.inspect_engine"(%a) {} : (!trt.engine) -> ()

  %res = "trt.compute"(%a, %gpu_ctx) {} : (!trt.engine, !phi.context<GPU>) -> (!infrt.tensor_list)
  %size = "dt.tensor_list_get_size"(%res) {} : (!infrt.tensor_list) -> (i32)
  "infrt.print.i32"(%size) {} : (i32) -> ()

  %ts0 = "dt.tensor_list_get_tensor"(%res) {id = 0 : i32} : (!infrt.tensor_list) -> (!infrt.dense_tensor<GPU, FP32, NCHW>)
  "phi_dt.print_tensor" (%ts0) : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> ()

  %ts1 = "dt.tensor_list_get_tensor"(%res) {id = 1 : i32} : (!infrt.tensor_list) -> (!infrt.dense_tensor<GPU, FP32, NCHW>)
  "phi_dt.print_tensor" (%ts1) : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> ()

  infrt.return
}

// CHECK-LABEL: @main
func @main() {
  %gpu_ctx = "phi_dt.create_context.gpu" (): () -> !phi.context<GPU>
  %cpu_ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>

  %input_tensor = "phi_dt.create_dense_tensor.gpu" (%gpu_ctx) {
    precision=#infrt.precision<FP32>,
    layout=#infrt.layout<NCHW>,
    dims=[1:i64, 3:i64, 28:i64, 28:i64], lod=[0:i64]}: (!phi.context<GPU>) -> (!infrt.dense_tensor<GPU, FP32, NCHW>)
  "phi_dt.fill_dense_tensor.f32"(%input_tensor) {value=[3.8:f32, 2.4:f32, 1.3:f32]} : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> ()
  // "phi_dt.print_tensor" (%input_tensor) : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> ()

  %kernel_weight = "phi_dt.create_dense_tensor.cpu"(%cpu_ctx) {
    precision=#infrt.precision<FP32>,
    layout=#infrt.layout<NCHW>,
    dims=[3:i64, 3:i64, 3:i64, 3:i64], lod=[0:i64]} : (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
  "phi_dt.fill_dense_tensor.f32"(%kernel_weight) {value=[1.:f32, 2.:f32, 3.:f32, 4.:f32, 5.:f32, 6.:f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
  // "phi_dt.print_tensor" (%kernel_weight) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()

  %kernel_bias = "phi_dt.create_dense_tensor.cpu"(%cpu_ctx) {
    precision=#infrt.precision<FP32>,
    layout=#infrt.layout<NCHW>,
    dims=[3:i64], lod=[0:i64]} : (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
  "phi_dt.fill_dense_tensor.f32"(%kernel_bias) {value=[1.:f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
  // "phi_dt.print_tensor" (%kernel_bias) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()

  infrt.call @run_trt(%input_tensor, %kernel_weight, %kernel_bias, %gpu_ctx) : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !phi.context<GPU>) -> ()

  infrt.return
}
