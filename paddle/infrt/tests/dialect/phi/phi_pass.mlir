// RUN: infrtopt -phi-op-convert=valid-targets=CPU-FP32-NCHW -infrt-op-fuse %s

// CHECK-LABEL: @ops
func @ops(%a:!infrt.dense_tensor<CPU, FP32, NCHW>, %b:!infrt.dense_tensor<CPU, FP32, NCHW>) {
  %g = "pd.elementwise_add"(%a, %b) {axis=1:si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
  %h = "pd.abs"(%g):(!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
  infrt.return %h:!infrt.dense_tensor<CPU, FP32, NCHW>
}

// CHECK-LABEL: @op_execute
func @op_execute(%a:!infrt.dense_tensor<CPU, FP32, NCHW>, %b:!infrt.dense_tensor<CPU, FP32, NCHW>, %c:!infrt.dense_tensor<CPU, FP32, NCHW>)  -> !infrt.dense_tensor<CPU, FP32, NCHW> {
  %g = "pd.elementwise_add"(%a, %b) {axis=1:si32} : (!infrt.dense_tensor<CPU, FP32, NCHW>, !infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
  %h = "pd.abs"(%g):(!infrt.dense_tensor<CPU, FP32, NCHW>) -> !infrt.dense_tensor<CPU, FP32, NCHW>
  infrt.return %h:!infrt.dense_tensor<CPU, FP32, NCHW>
}
