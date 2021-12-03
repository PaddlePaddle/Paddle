// CHECK-LABEL: @fc
func @fc(%input : !infrt.tensor<X86, NCHW, F32>,
         %w : !infrt.tensor<X86, NCHW, F32>,
         %bias : !infrt.tensor<X86, NCHW, F32>) -> !infrt.tensor<X86, NCHW, F32>
{
  %out = dt.create_uninit_tensor.f32 [30, 50] -> !infrt.tensor<X86, NCHW, F32>
  // dt.fill_tensor_with_constant.f32 (%out : !infrt.tensor<X86, NCHW, F32>) {value=0.0:f32}

  // fc1
  "external.matmul"(%input, %w, %out) {}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()

  // fc2
  "external.matmul"(%out, %w, %out) {}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()

  infrt.return %out : !infrt.tensor<X86, NCHW, F32>
}

// CHECK-LABEL: @benchmark
func @benchmark() {
  %input = dt.create_uninit_tensor.f32 [30, 50] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%input : !infrt.tensor<X86, NCHW, F32>) {value=1.0:f32}

  %w = dt.create_uninit_tensor.f32 [50, 50] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%w : !infrt.tensor<X86, NCHW, F32>) {value=2.0:f32}

  %bias = dt.create_uninit_tensor.f32 [30, 50] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%bias : !infrt.tensor<X86, NCHW, F32>) {value=3.0:f32}

  infrt.benchmark "add.f32"(
          %input:!infrt.tensor<X86, NCHW, F32>,
          %w:!infrt.tensor<X86, NCHW, F32>,
          %bias:!infrt.tensor<X86, NCHW, F32>)
          duration_secs = 100, max_count = 300000, num_warmup_runs = 3
  {
    %res = infrt.call @fc(%input, %w, %bias) : (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> (!infrt.tensor<X86, NCHW, F32>)
    infrt.return %res : !infrt.tensor<X86, NCHW, F32>
  }
  infrt.return
}
