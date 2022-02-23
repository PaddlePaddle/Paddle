// CHECK-LABEL: @fc
func @fc(%input : !Infrt.tensor<X86, NCHW, F32>,
         %w : !Infrt.tensor<X86, NCHW, F32>,
         %bias : !Infrt.tensor<X86, NCHW, F32>) -> !Infrt.tensor<X86, NCHW, F32>
{
  %out = dt.create_uninit_tensor.f32 [30, 50] -> !Infrt.tensor<X86, NCHW, F32>
  // dt.fill_tensor_with_constant.f32 (%out : !Infrt.tensor<X86, NCHW, F32>) {value=0.0:f32}

  // fc1
  "external.matmul"(%input, %w, %out) {}: (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> ()

  // fc2
  "external.matmul"(%out, %w, %out) {}: (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> ()
  "external.elementwise_add"(%out, %bias, %out) {axis = -1}: (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> ()
  "external.sigmoid"(%out, %out) {}: (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> ()

  Infrt.return %out : !Infrt.tensor<X86, NCHW, F32>
}

// CHECK-LABEL: @benchmark
func @benchmark() {
  %input = dt.create_uninit_tensor.f32 [30, 50] -> !Infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%input : !Infrt.tensor<X86, NCHW, F32>) {value=1.0:f32}

  %w = dt.create_uninit_tensor.f32 [50, 50] -> !Infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%w : !Infrt.tensor<X86, NCHW, F32>) {value=2.0:f32}

  %bias = dt.create_uninit_tensor.f32 [30, 50] -> !Infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%bias : !Infrt.tensor<X86, NCHW, F32>) {value=3.0:f32}

  Infrt.benchmark "add.f32"(
          %input:!Infrt.tensor<X86, NCHW, F32>,
          %w:!Infrt.tensor<X86, NCHW, F32>,
          %bias:!Infrt.tensor<X86, NCHW, F32>)
          duration_secs = 100, max_count = 300000, num_warmup_runs = 3
  {
    %res = Infrt.call @fc(%input, %w, %bias) : (!Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>, !Infrt.tensor<X86, NCHW, F32>) -> (!Infrt.tensor<X86, NCHW, F32>)
    Infrt.return %res : !Infrt.tensor<X86, NCHW, F32>
  }
  Infrt.return
}
