// CHECK: paddle_func
func @paddle_func() -> () {
  %input = dt.create_uninit_tensor.f32 [3, 5] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%input : !infrt.tensor<X86, NCHW, F32>) {value=1.0:f32}

  %w = dt.create_uninit_tensor.f32 [5, 4] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%w : !infrt.tensor<X86, NCHW, F32>) {value=2.0:f32}

  %bias = dt.create_uninit_tensor.f32 [4] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%bias : !infrt.tensor<X86, NCHW, F32>) {value=3.0:f32}

  %out = dt.create_uninit_tensor.f32 [3, 4] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%out : !infrt.tensor<X86, NCHW, F32>) {value=0.0:f32}

  "external.fc2"(%input, %w, %bias, %out) {in_num_col_dims=3:i32, test_attr=5:i32}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  // CHECK-LABEL: tensor: shape=shape[3,5], values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  dt.print_tensor (%input : !infrt.tensor<X86, NCHW, F32>)
  // CHECK-LABEL: tensor: shape=shape[5,4], values=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  dt.print_tensor (%w : !infrt.tensor<X86, NCHW, F32>)
  dt.print_tensor (%bias : !infrt.tensor<X86, NCHW, F32>)
  dt.print_tensor (%out : !infrt.tensor<X86, NCHW, F32>)

  // test external.matmul
  %out1 = dt.create_uninit_tensor.f32 [3, 4] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%out1 : !infrt.tensor<X86, NCHW, F32>) {value=0.0:f32}
  "external.matmul"(%input, %w, %out1) {}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  dt.print_tensor (%out1 : !infrt.tensor<X86, NCHW, F32>)

  // test external.elementwise_add
  %out2 = dt.create_uninit_tensor.f32 [3, 4] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%out2 : !infrt.tensor<X86, NCHW, F32>) {value=0.0:f32}
  %bias1 = dt.create_uninit_tensor.f32 [3, 4] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%bias1 : !infrt.tensor<X86, NCHW, F32>) {value=3.0:f32}
  "external.elementwise_add"(%out1, %bias1, %out2) {axis=-1}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  dt.print_tensor (%out2 : !infrt.tensor<X86, NCHW, F32>)

  // test external.relu
  %out3 = dt.create_uninit_tensor.f32 [3, 4] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%out3 : !infrt.tensor<X86, NCHW, F32>) {value=0.0:f32}
  "external.relu"(%out1, %out3) {}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  dt.print_tensor (%out3 : !infrt.tensor<X86, NCHW, F32>)

  // test external.sigmoid
  %out4 = dt.create_uninit_tensor.f32 [3, 4] -> !infrt.tensor<X86, NCHW, F32>
  dt.fill_tensor_with_constant.f32 (%out4 : !infrt.tensor<X86, NCHW, F32>) {value=0.0:f32}
  "external.sigmoid"(%out1, %out4) {}: (!infrt.tensor<X86, NCHW, F32>, !infrt.tensor<X86, NCHW, F32>) -> ()
  dt.print_tensor (%out4 : !infrt.tensor<X86, NCHW, F32>)

  infrt.return
}
