module {
  func.func @test_add(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = "tosa.const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %1 = "tosa.const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %2 = "tosa.const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %3 = "tosa.const"() {value = dense<4.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %4 = "tosa.const"() {value = dense<5.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %5 = "tosa.const"() {value = dense<6.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %6 = "tosa.const"() {value = dense<7.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %7 = "tosa.const"() {value = dense<8.000000e+00> : tensor<f32>} : () -> tensor<f32>
    %8 = "tosa.const"() {value = dense<9.000000e+00> : tensor<f32>} : () -> tensor<f32>

    %9 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %10 = "tosa.const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
    %11:3 = "tosa.while_loop"(%9, %10, %arg0) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<10xf32>):
      %21 = "tosa.greater_equal"(%arg2, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "tosa.yield"(%21) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<10xf32>):
      %21 = "tosa.add"(%arg3, %0) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
      %22 = "tosa.add"(%21, %1) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
      %23 = "tosa.add"(%22, %2) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
      %24 = "tosa.add"(%23, %3) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
      %25 = "tosa.add"(%24, %4) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
      %26 = "tosa.add"(%25, %5) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
      %27 = "tosa.add"(%26, %6) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
      %28 = "tosa.add"(%27, %7) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
      "tosa.yield"(%arg1, %arg2, %28) : (tensor<i32>, tensor<i32>, tensor<10xf32>) -> ()
    }) : (tensor<i32>, tensor<i32>, tensor<10xf32>) -> (tensor<i32>, tensor<i32>, tensor<10xf32>)

    %12 = "tosa.add"(%11#2, %0) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    %13 = "tosa.add"(%12, %1) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    %14 = "tosa.add"(%13, %2) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    %15 = "tosa.add"(%14, %3) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    %16 = "tosa.add"(%15, %4) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    %17 = "tosa.add"(%16, %5) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    %18 = "tosa.add"(%17, %6) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    %19 = "tosa.add"(%18, %7) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    %20 = "tosa.add"(%19, %8) : (tensor<10xf32>, tensor<f32>) -> tensor<10xf32>
    return %20 : tensor<10xf32>
  }
}
