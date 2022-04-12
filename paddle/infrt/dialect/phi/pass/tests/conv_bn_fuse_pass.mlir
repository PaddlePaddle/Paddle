// RUN: infrtopt --conv2d-bn-fuse %s | FileCheck %s

module  {
  func @main_graph(%arg0: !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW> {
    %0 = "phi_dt.create_context.cpu"() : () -> !phi.context<CPU>
    %1 = "phi_dt.create_context.cpu"() : () -> !phi.context<CPU>
    // CHECK-LABEL: %1 = "phi_dt.create_host_inited_dense_tensor.f32"(%0) {dims = [8, 3, 1, 1], layout = #infrt.layout<NCHW>, lod = [0], run_once = true, values = [0.999994993 : f32, 9.999950e+00 : f32, 99.9994964 : f32, 31.6227627 : f32, 316.227631 : f32, 3.16227627 : f32, 1.000000e+02 : f32, 1.000000e+03 : f32, 1.000000e+02 : f32, 316.227631 : f32, 3.16227627 : f32, 31.6227627 : f32, 1.000000e+03 : f32, 1.000000e+02 : f32, 1.000000e+03 : f32, 0.999994993 : f32, 9.999950e+00 : f32, 99.9994964 : f32, 9.999950e+00 : f32, 99.9994964 : f32, 0.999994993 : f32, 0.999994993 : f32, 0.999994993 : f32, 0.999994993 : f32]} : (!phi.context<CPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %2 = "phi_dt.create_host_inited_dense_tensor.f32"(%0) {run_once = true, dims = [8], layout = #infrt.layout<NCHW>, lod = [0], values = [1.0 : f32, 10.0 : f32, 100.0 : f32, 10.0 : f32, 100.0 : f32, 1.0 : f32, 1.0 : f32, 1.0 : f32]} : (!phi.context<CPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %3 = "phi_dt.create_host_inited_dense_tensor.f32"(%0) {run_once = true, dims = [8, 3, 1, 1], layout = #infrt.layout<NCHW>, lod = [0], values = [1.0 : f32, 10.0 : f32, 100.0 : f32, 10.0 : f32, 100.0 : f32, 1.0 : f32, 10.0 : f32, 100.0 : f32, 10.0 : f32, 100.0 : f32, 1.0 : f32, 10.0 : f32, 100.0 : f32, 10.0 : f32, 100.0 : f32, 1.0 : f32, 10.0 : f32, 100.0 : f32, 10.0 : f32, 100.0 : f32, 1.0 : f32, 1.0 : f32, 1.0 : f32, 1.0 : f32]} : (!phi.context<CPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    // CHECK-LABEL: %4 = "phi_dt.create_host_inited_dense_tensor.f32"(%3) {dims = [8], layout = #infrt.layout<NCHW>, lod = [0], run_once = true, values = [5.006790e-06 : f32, -21.6227627 : f32, -9.000000e+02 : f32, -21.6227627 : f32, -9.000000e+02 : f32, 5.006790e-06 : f32, 5.006790e-06 : f32, 5.006790e-06 : f32]} : (!phi.context<CPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %4 = "phi_dt.create_host_inited_dense_tensor.f32"(%0) {run_once = true, dims = [8], layout = #infrt.layout<NCHW>, lod = [0], values = [1.0 : f32, 10.0 : f32, 100.0 : f32, 10.0 : f32, 100.0 : f32, 1.0 : f32, 1.0 : f32, 1.0 : f32]} : (!phi.context<CPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    // CHECK-LABEL: %5 = "pd.elementwise_add"(%2, %4) {axis = 1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %5 = "phi_dt.create_host_inited_dense_tensor.f32"(%0) {run_once = true, dims = [8], layout = #infrt.layout<NCHW>, lod = [0], values = [1.0 : f32, 10.0 : f32, 100.0 : f32, 10.0 : f32, 100.0 : f32, 1.0 : f32, 1.0 : f32, 1.0 : f32]} : (!phi.context<CPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %6 = "phi_dt.create_host_inited_dense_tensor.f32"(%0) {run_once = true, dims = [8], layout = #infrt.layout<NCHW>, lod = [0], values = [1.0 : f32, 10.0 : f32, 100.0 : f32, 10.0 : f32, 100.0 : f32, 1.0 : f32, 1.0 : f32, 1.0 : f32]} : (!phi.context<CPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %7 = "pd.conv2d"(%arg0, %3) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [3 : i32, 3 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %8 = "pd.batch_norm"(%7, %2, %6, %4, %5) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    infrt.return %8 : !infrt.dense_tensor<CPU, FP32, NCHW>
  }

  func @main() {
    %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
    %input_tensor = "phi_dt.create_dense_tensor.cpu" (%ctx) {
      precision=#infrt.precision<FP32>,
      layout=#infrt.layout<NCHW>,
      dims=[1:i64, 3:i64, 1:i64, 1:i64], lod=[1:i64]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    "phi_dt.fill_dense_tensor.f32"(%input_tensor) {value=[1.0 : f32, 1.0 : f32, 1.0 : f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
    %res = infrt.call @main_graph(%input_tensor) {} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    "phi_dt.print_tensor" (%res) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
    infrt.return
  }
}
