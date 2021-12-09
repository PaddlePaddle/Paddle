// CHECK-LABEL: @main
func @main() -> tensor<?xf32> {
  %a = "pd.Feed"() : () -> tensor<?xf32>
  %b = "pd.Feed"() : () -> tensor<?xf32>
  %bias = "pd.Feed"() : () -> tensor<?xf32>

  %b1 = "pd.Feed"() : () -> tensor<?xf32>
  %b2 = "pd.Feed"() : () -> tensor<?xf32>
  %bias1 = "pd.Feed"() : () -> tensor<?xf32>
  %bias2 = "pd.Feed"() : () -> tensor<?xf32>

  %c = "pd.Matmul"(%a, %b) {transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %d = "pd.ElementwiseAdd"(%c, %bias) {axis=1:i32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e = "pd.Relu6"(%d) {} : (tensor<?xf32>) -> tensor<?xf32>

  %c1 = "pd.Matmul"(%e, %b1) {transpose_x=false, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %d1 = "pd.ElementwiseAdd"(%c1, %bias1) {axis=1:i32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e1 = "pd.Relu"(%d1) {} : (tensor<?xf32>) -> tensor<?xf32>

  %c2 = "pd.Matmul"(%e1, %b2) {transpose_x=true, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %d2 = "pd.ElementwiseAdd"(%c2, %bias2) {axis=1:i32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e2 = "pd.Relu"(%d2) {} : (tensor<?xf32>) -> tensor<?xf32>
  infrt.return %e2 : tensor<?xf32>
}