func @load_tensor_map() {
  %path = infrt.get_string("/shixiaowei02/Paddle-Compute-Library/Paddle-InfRT/Paddle/build_ubuntu/multi_fc_model")
  %map = dt.load_params(%path)
  %size = dt.tensor_map_get_size(%map) -> i32
  // CHECK: 7
  infrt.print.i32 %size

  %tensor_name = infrt.get_string("fc_bias")
  %a = dt.tensor_map_get_tensor(%map, %tensor_name) -> !infrt.tensor<X86, NCHW, F32>

  dt.print_tensor (%a : !infrt.tensor<X86, NCHW, F32>)

  infrt.return
}
