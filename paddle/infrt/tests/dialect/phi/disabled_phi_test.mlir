// RUN: infrtexec -i %s
module  {
  func @predict(%arg0: !infrt.dense_tensor<CPU, FP32, NCHW>,%filter: !infrt.dense_tensor<CPU, FP32, NCHW>, %arg1: !infrt.dense_tensor<CPU, FP32, NCHW>, %arg2: !infrt.dense_tensor<CPU, FP32, NCHW>, %arg3: !infrt.dense_tensor<CPU, FP32, NCHW>, %arg4: !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW> {
    %2 = "pd.abs"(%arg0) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %3 = "pd.matmul_v2"(%arg0, %2) {trans_x = false, trans_y = false} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %4 = "pd.conv2d"(%3, %filter) {data_format = "NCHW", dilations = [1 : i32, 1 : i32], groups = 1 : si32, padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y, %MeanOut, %VarianceOut = "pd.batch_norm"(%4, %arg1, %arg2, %arg3, %arg4) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    %out = "pd.relu"(%Y) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %5 = "pd.elementwise_add"(%out, %out) {axis = -1:si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %6 = "pd.pool2d"(%5) {adaptive = false, pooling_type = "avg", ceil_mode = false, data_format = "NCHW", exclusive = true, global_pooling = false, ksize = [3 : i32, 3 : i32], padding_algorithm = "EXPLICIT", paddings = [1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %7 = "pd.flatten_contiguous_range"(%6) {start_axis = 1 : si32, stop_axis = 3 : si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    infrt.return %7 : !infrt.dense_tensor<CPU, FP32, NCHW>
  }
  func @main() {
    %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
    %t = "phi_dt.create_inited_dense_tensor.cpu.f32"(%ctx) {value=3.8:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[1, 3, 8, 8]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %filter = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=3.8:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[3, 3, 8, 8]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %bias = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=1.5:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %mean = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=3.8:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %scale = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=3.8:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %var = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=3.8:f32, layout=#infrt.layout<NCHW>, lod=[1], dims=[3]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    
    %2 = infrt.call@predict(%t, %filter, %bias, %mean, %scale, %var) : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>,!infrt.dense_tensor<CPU, FP32, NCHW>,!infrt.dense_tensor<CPU, FP32, NCHW>,!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    phi_dt.print_tensor(%2 : !infrt.dense_tensor<CPU, FP32, NCHW>)
    infrt.return
  }
}
