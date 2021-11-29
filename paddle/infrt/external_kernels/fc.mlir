// CHECK-LABEL: @fc
func @fc(%input : !cinn.tensor<X86, NCHW, F32>,
         %w : !cinn.tensor<X86, NCHW, F32>,
         %bias : !cinn.tensor<X86, NCHW, F32>) -> !cinn.tensor<X86, NCHW, F32>
{
  %out = dt.create_uninit_tensor.f32 [30, 50] -> !cinn.tensor<X86, NCHW, F32>
  // dt.fill_tensor_with_constant.f32 (%out : !cinn.tensor<X86, NCHW, F32>) {value=0.0:f32}

  // fc1
  "external.matmul"(%input, %w, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()

  // fc2
  "external.matmul"(%out, %w, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> ()

  cinn.return %out : !cinn.tensor<X86, NCHW, F32>
}

// CHECK-LABEL: @benchmark
func @benchmark() {
  %input = dt.create_uninit_tensor.f32 [30, 50] -> !cinn.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%input : !cinn.tensor<X86, NCHW, F32>) {value=1.0:f32}

  %w = dt.create_uninit_tensor.f32 [50, 50] -> !cinn.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%w : !cinn.tensor<X86, NCHW, F32>) {value=2.0:f32}

  %bias = dt.create_uninit_tensor.f32 [30, 50] -> !cinn.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%bias : !cinn.tensor<X86, NCHW, F32>) {value=3.0:f32}

  cinn.benchmark "add.f32"(
          %input:!cinn.tensor<X86, NCHW, F32>,
          %w:!cinn.tensor<X86, NCHW, F32>,
          %bias:!cinn.tensor<X86, NCHW, F32>)
          duration_secs = 100, max_count = 300000, num_warmup_runs = 3
  {
    %res = cinn.call @fc(%input, %w, %bias) : (!cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>, !cinn.tensor<X86, NCHW, F32>) -> (!cinn.tensor<X86, NCHW, F32>)
    cinn.return %res : !cinn.tensor<X86, NCHW, F32>
  }
  cinn.return
}
