// RUN: infrtexec -i %s | FileCheck %s
module  {
  func @main() {
    %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
    %0 = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value = 2.0 : f32, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[1, 3, 6, 6]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %1 = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value = 2.0 : f32, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[1, 3, 3, 3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    
    %2 = "pd.conv2d"(%0, %1) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [3 : i32, 3 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    // CHECK: dense_tensor: shape=shape[1, 1, 5, 5], value=[0,0,0,0,0,0,48,72,72,24,0,72,108,108,36,0,72,108,108,36,0,24,36,36,12]
    phi_dt.print_tensor (%2 : !infrt.dense_tensor<CPU, FP32, NCHW>)
    
    %3 = "pd.relu"(%2) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    // dense_tensor: shape=shape[1, 1, 5, 5], value=[0,0,0,0,0,0,48,72,72,24,0,72,108,108,36,0,72,108,108,36,0,24,36,36,12]
    phi_dt.print_tensor (%3 : !infrt.dense_tensor<CPU, FP32, NCHW>)

    %4 = "pd.pool2d"(%2) {adaptive = false, ceil_mode = false, data_format = "NCHW", exclusive = true, global_pooling = false, ksize = [2 : i32, 2 : i32], padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], pooling_type = "avg", strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    // CHECK: dense_tensor: shape=shape[1, 1, 3, 3], value=[0,0,0,0,75,60,0,60,48]
    phi_dt.print_tensor (%4 : !infrt.dense_tensor<CPU, FP32, NCHW>)

    %5 = "pd.flatten_contiguous_range"(%4) {start_axis = 1 : si32, stop_axis = 3 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    // CHECK: dense_tensor: shape=shape[1, 9], value=[0,0,0,0,75,60,0,60,48]
    phi_dt.print_tensor (%5 : !infrt.dense_tensor<CPU, FP32, NCHW>)
    
    %6 = "pd.elementwise_add"(%5, %5) {axis = 1 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW> 
    // CHECK: dense_tensor: shape=shape[1, 9], value=[0,0,0,0,150,120,0,120,96]
    phi_dt.print_tensor (%6 : !infrt.dense_tensor<CPU, FP32, NCHW>)
    
    %7 = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value = 4.0 : f32, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[9, 3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %8 = "pd.matmul_v2"(%5, %7) {trans_x = false, trans_y = false} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    // CHECK: dense_tensor: shape=shape[1, 3], value=[972,972,972]
    phi_dt.print_tensor (%8 : !infrt.dense_tensor<CPU, FP32, NCHW>)

    %scale = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=1.0:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %bias = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=1.8:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %mean = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=2.0:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %var = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=0.0:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %Y, %MeanOut, %VarianceOut = "pd.batch_norm"(%1, %scale, %bias, %mean, %var) {data_layout = "NCHW", epsilon = 0.01 : f32, momentum = 0.5 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    // CHECK: dense_tensor: shape=shape[1, 3, 3, 3], value=[1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8]
    phi_dt.print_tensor (%Y : !infrt.dense_tensor<CPU, FP32, NCHW>)
  
    infrt.return
  }
}
