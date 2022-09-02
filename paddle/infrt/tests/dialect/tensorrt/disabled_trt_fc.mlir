module  {
  func @main_graph(%arg0: !infrt.dense_tensor<CPU, FP32, ANY>) -> !infrt.dense_tensor<CPU, FP32, ANY> {
    %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
    %0 = "phi_dt.create_context.gpu"() : () -> !phi.context<GPU>
    %1 = "phi_dt.memcpy.gpu"(%arg0, %0) {d2h = false} : (!infrt.dense_tensor<CPU, FP32, ANY>, !phi.context<GPU>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %4 = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=1.5:f32, layout=#infrt.layout<NCHW>, lod=[0], dims=[2, 6]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %3 = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=1.5:f32, layout=#infrt.layout<NCHW>, lod=[0], dims=[2]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %5 = "trt.create_engine"(%1, %4, %3) ( {
      %10 = "trt.FullyConnected"(%1, %4, %3) {out_channel_num = 2 : si32} : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
      infrt.return %10 : !infrt.dense_tensor<GPU, FP32, NCHW>
    }) {run_once = true} : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !trt.engine
    %6 = "trt.compute"(%5, %0) : (!trt.engine, !phi.context<GPU>) -> !infrt.tensor_list
    %7 = "dt.tensor_list_get_tensor"(%6) {id = 0 : i32} : (!infrt.tensor_list) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %8 = "phi_dt.memcpy.gpu"(%7, %0) {d2h = true} : (!infrt.dense_tensor<GPU, FP32, NCHW>, !phi.context<GPU>) -> !infrt.dense_tensor<CPU, FP32, ANY>
    infrt.return %8 : !infrt.dense_tensor<CPU, FP32, ANY>
  }

  func @main() {
    %ctx = "phi_dt.create_context.cpu" (): () -> !phi.context<CPU>
    %input_tensor = "phi_dt.create_inited_dense_tensor.cpu.f32" (%ctx) {value=1.5:f32, layout=#infrt.layout<NCHW>, lod=[0], dims=[3, 6, 1, 1]}: (!phi.context<CPU>) -> (!infrt.dense_tensor<CPU, FP32, NCHW>)
    %res = infrt.call @main_graph(%input_tensor) {} : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
    "phi_dt.print_tensor" (%res) : (!infrt.dense_tensor<CPU, FP32, NCHW>) -> ()
    infrt.return
  }
}
