module  {
  func @main_graph(%arg0: !phi.dense_tensor_map,  %arg1: !infrt.dense_tensor<GPU, FP32, NCHW>, %2: !phi.context<GPU>) -> !infrt.dense_tensor<GPU, FP32, NCHW> {
    %0 = phi_dt.tensor_map_get_tensor(%arg0) {name = "linear_0.w_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %1 = phi_dt.tensor_map_get_tensor(%arg0) {name = "linear_0.b_0"} -> !infrt.dense_tensor<CPU, FP32, NCHW>
    %3 = "phi_dt.memcpy.gpu"(%0, %2) {d2h = false} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !phi.context<GPU>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %4 = "phi_gpu.matmul.float32.any"(%2, %arg1, %3) {trans_x = false, trans_y = false} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %5 = "phi_dt.memcpy.gpu"(%1, %2) {d2h = false} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !phi.context<GPU>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %6 = "phi_gpu.add_raw.float32.any"(%2, %4, %5) {axis = 1 : si32} : (!phi.context<GPU>, !infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    infrt.return %6 : !infrt.dense_tensor<GPU, FP32, NCHW>
  }
  func @main() {
    %0 = phi_dt.load_combined_params() {model_path = "linear/linear.pdmodel", params_path = "linear/linear.pdiparams"}
    %1 = "phi_dt.create_context.cpu"() : () -> !phi.context<CPU>
    %2 = "phi_dt.create_dense_tensor.cpu"(%1) {dims = [16, 784], layout = #infrt.layout<NCHW>, lod = [1], precision = #infrt.precision<FP32>} : (!phi.context<CPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>

    %4 = "phi_dt.create_context.gpu"() : () -> !phi.context<GPU>
    %5 = "phi_dt.memcpy.gpu"(%2, %4) {d2h = false}:(!infrt.dense_tensor<CPU, FP32, NCHW>, !phi.context<GPU>) -> !infrt.dense_tensor<GPU, FP32, NCHW>

    phi_dt.fill_dense_tensor.f32(%2 : !infrt.dense_tensor<CPU, FP32, NCHW>) {value = [3.800000e+00 : f32, 2.400000e+00 : f32, 1.300000e+00 : f32]}
    %6 = infrt.call @main_graph(%0, %5, %4) : (!phi.dense_tensor_map, !infrt.dense_tensor<GPU, FP32, NCHW>, !phi.context<GPU>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
    %7 = "phi_dt.memcpy.gpu"(%6, %1) {d2h = true}:(!infrt.dense_tensor<GPU, FP32, NCHW>, !phi.context<CPU>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
//    phi_dt.print_tensor(%7 : !infrt.dense_tensor<CPU, FP32, NCHW>)
    infrt.return
  }
}
