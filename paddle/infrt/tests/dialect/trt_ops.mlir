// RUN: trt-exec %s
// CHECK-LABEL: @main
func @main(%bias:!infrt.dense_tensor<GPU, FP32, NCHW>, %c:!infrt.dense_tensor<GPU, FP32, NCHW>, %b1:!infrt.dense_tensor<GPU, FP32, NCHW>, %b2:!infrt.dense_tensor<GPU, FP32, NCHW>, %bias1:!infrt.dense_tensor<GPU, FP32, NCHW>, %bias2:!infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW> {
  %d = "pd.elementwise_add"(%c, %bias) {axis=-1:si32} : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
  %e = "pd.relu6"(%d) {} : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>

  %c1 = "pd.matmul"(%e, %b1) {transpose_x=false, transpose_y=false} : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
  %d1 = "pd.elementwise_add"(%c1, %bias1) {axis=-1:si32} : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
  %e1 = "pd.relu"(%d1) {} : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>

  %c2 = "pd.matmul"(%e1, %b2) {transpose_x=true, transpose_y=false} : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
  %d2 = "pd.elementwise_add"(%c2, %bias2) {axis=-1:si32} : (!infrt.dense_tensor<GPU, FP32, NCHW>, !infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
  %e2 = "pd.relu"(%d2) {} : (!infrt.dense_tensor<GPU, FP32, NCHW>) -> !infrt.dense_tensor<GPU, FP32, NCHW>
  
  infrt.return %e2 : !infrt.dense_tensor<GPU, FP32, NCHW>
}
