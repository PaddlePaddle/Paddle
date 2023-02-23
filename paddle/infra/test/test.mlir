// void func(tensor a_tensor, int num) {
//   int i = 0;
//   while (i++ < num) {
//     auto i_tensor = make_tensor(i, a.shape);
//     if (i >= 5) {
//       a_tensor += i_tensor;
//     } else {
//       a_tensor -= i_tensor;
//     }
//   }
// }

module {
  func.func @test_while_loop(%arg0: tensor<10xi32>, %arg1: tensor<i32>) {
    %0 = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tosa.const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
    %2:3 = "tosa.while_loop"(%0, %0, %arg0) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<10xi32>):
      %3 = "tosa.greater_equal"(%arg3, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4 = "tosa.logical_not"(%3) : (tensor<i1>) -> tensor<i1>
      "tosa.yield"(%4) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<10xi32>):
      %3 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %4 = "tosa.add"(%arg3, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %5 = "tosa.add"(%arg2, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %6 = "tosa.reshape"(%3) {new_shape = array<i64: 1>} : (tensor<i32>) -> tensor<1xi32>
      %7 = "tosa.greater_equal"(%arg2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %8 = "tosa.cond_if"(%7, %arg4, %6) ({
      ^bb0(%arg5: tensor<10xi32>, %arg6: tensor<1xi32>):
        %9 = "tosa.add"(%arg5, %arg6) : (tensor<10xi32>, tensor<1xi32>) -> tensor<10xi32>
        "tosa.yield"(%9) : (tensor<10xi32>) -> ()
      }, {
      ^bb0(%arg5: tensor<10xi32>, %arg6: tensor<1xi32>):
        %9 = "tosa.sub"(%arg5, %arg6) : (tensor<10xi32>, tensor<1xi32>) -> tensor<10xi32>
        "tosa.yield"(%9) : (tensor<10xi32>) -> ()
      }) : (tensor<i1>, tensor<10xi32>, tensor<1xi32>) -> tensor<10xi32>
      "tosa.yield"(%5, %4, %8) : (tensor<i32>, tensor<i32>, tensor<10xi32>) -> ()
    }) : (tensor<i32>, tensor<i32>, tensor<10xi32>) -> (tensor<i32>, tensor<i32>, tensor<10xi32>)
    return
  }
}
