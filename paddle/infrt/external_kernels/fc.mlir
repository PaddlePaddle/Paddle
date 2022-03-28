// CHECK-LABEL: @fc
func @fc(%input : !core.dense_tensor<CPU, FP32, NCHW>,
         %w : !core.dense_tensor<CPU, FP32, NCHW>,
         %bias : !core.dense_tensor<CPU, FP32, NCHW>) -> !core.dense_tensor<CPU, FP32, NCHW>
{
  %out = dt.create_uninit_tensor.f32 [30, 50] -> !core.dense_tensor<CPU, FP32, NCHW>
  // dt.fill_tensor_with_constant.f32 (%out : !core.dense_tensor<CPU, FP32, NCHW>) {value=0.0:f32}

  // fc1
  "external.matmul"(%input, %w, %out) {}: (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> ()
  "external.sigmoid"(%out, %out) {}: (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> ()

  // fc2
  "external.matmul"(%out, %w, %out) {}: (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> ()
  "external.sigmoid"(%out, %out) {}: (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> ()

  core.return %out : !core.dense_tensor<CPU, FP32, NCHW>
}

// CHECK-LABEL: @benchmark
func @benchmark() {
  %input = dt.create_uninit_tensor.f32 [30, 50] -> !core.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%input : !core.dense_tensor<CPU, FP32, NCHW>) {value=1.0:f32}

  %w = dt.create_uninit_tensor.f32 [50, 50] -> !core.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%w : !core.dense_tensor<CPU, FP32, NCHW>) {value=2.0:f32}

  %bias = dt.create_uninit_tensor.f32 [30, 50] -> !core.dense_tensor<CPU, FP32, NCHW>
  dt.fill_tensor_with_constant.f32 (%bias : !core.dense_tensor<CPU, FP32, NCHW>) {value=3.0:f32}

  core.benchmark "add.f32"(
          %input:!core.dense_tensor<CPU, FP32, NCHW>,
          %w:!core.dense_tensor<CPU, FP32, NCHW>,
          %bias:!core.dense_tensor<CPU, FP32, NCHW>)
          duration_secs = 100, max_count = 300000, num_warmup_runs = 3
  {
    %res = core.call @fc(%input, %w, %bias) : (!core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>, !core.dense_tensor<CPU, FP32, NCHW>) -> (!core.dense_tensor<CPU, FP32, NCHW>)
    core.return %res : !core.dense_tensor<CPU, FP32, NCHW>
  }
  core.return
}
