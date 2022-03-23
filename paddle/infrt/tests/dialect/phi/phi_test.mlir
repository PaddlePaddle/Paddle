// RUN: infrtexec -i %s
module  {
  func @predict(%arg0: !infrt.dense_tensor<CPU, FP32, NCHW>, %arg1: !infrt.dense_tensor<CPU, FP32, NCHW>, %arg2: !infrt.dense_tensor<CPU, FP32, NCHW>, %arg3: !infrt.dense_tensor<CPU, FP32, NCHW>, %arg4: !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW> {
    %2 = "pd.abs"(%arg0) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %3 = "pd.matmul_v2"(%arg0, %2) {trans_x = false, trans_y = false} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %Y, %MeanOut, %VarianceOut = "pd.batch_norm"(%3, %arg1, %arg2, %arg3, %arg4) {data_layout = "NCHW", epsilon = 9.99999974E-6 : f32, momentum = 0.899999976 : f32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>)
    infrt.return %Y : !infrt.dense_tensor<CPU, FP32, NCHW>
  }
  func @main() {
    %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
    %t = "phi_dt.create_dense_tensor.cpu" (%ctx) {precision=#infrt.precision<FP32>, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[1:i64, 3:i64, 8:i64, 8:i64]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    "phi_dt.fill_dense_tensor.f32"(%t) {value=[3.8:f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
    %bias = "phi_dt.create_dense_tensor.cpu" (%ctx) {precision=#infrt.precision<FP32>, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[3:i64]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    "phi_dt.fill_dense_tensor.f32"(%bias) {value=[1.5:f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
    %mean = "phi_dt.create_dense_tensor.cpu" (%ctx) {precision=#infrt.precision<FP32>, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[3:i64]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    "phi_dt.fill_dense_tensor.f32"(%mean) {value=[3.5:f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
    %scale = "phi_dt.create_dense_tensor.cpu" (%ctx) {precision=#infrt.precision<FP32>, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[3:i64]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    "phi_dt.fill_dense_tensor.f32"(%scale) {value=[1.0:f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
    %var = "phi_dt.create_dense_tensor.cpu" (%ctx) {precision=#infrt.precision<FP32>, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[3:i64]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    "phi_dt.fill_dense_tensor.f32"(%var) {value=[0.0:f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
    
    %2 = infrt.call@predict(%t, %bias, %mean, %scale, %var) : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>,!infrt.dense_tensor<CPU, FP32, NCHW>,!infrt.dense_tensor<CPU, FP32, NCHW>,!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    
    //phi_dt.print_tensor(%t : !infrt.dense_tensor<CPU, FP32, NCHW>)
    phi_dt.print_tensor(%2 : !infrt.dense_tensor<CPU, FP32, NCHW>)
    infrt.return
  }
}
