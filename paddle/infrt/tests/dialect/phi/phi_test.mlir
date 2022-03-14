// RUN: infrtexec -i %s
module  {
  func @predict(%arg0: !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW> {
    %2 = "pd.abs"(%arg0) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    Infrt.return %2 : !infrt.dense_tensor<CPU, FP32, NCHW>
  }
  func @main() {
    %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
    %t = "phi_dt.create_dense_tensor" (%ctx) {precision=#infrt.precision<FP32>, layout=#infrt.layout<NCHW>, lod=[1:i64], dims=[1:i64]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    "phi_dt.fill_dense_tensor.f32"(%t) {value=[3.8:f32]} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
    %2 = Infrt.call@predict(%t) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    phi_dt.print_tensor(%2 : !infrt.dense_tensor<CPU, FP32, NCHW>)
    Infrt.return
  }
}
