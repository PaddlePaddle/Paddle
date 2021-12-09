func @build_tensor1() {
  %a = ts.build_shape [1:i64, 57:i64, 92:i64]
  ts.print_shape %a
  infrt.return
}
