// CHECK-LABEL: @main
func @main() -> tensor<?xf32> {
  %bias = "pd.feed"() {name="input0"} : () -> tensor<?xf32>
  %c = "pd.feed"() {name="input1"} : () -> tensor<?xf32>
  %b1 = "pd.feed"() {name="input2"} : () -> tensor<?xf32>
  %b2 = "pd.feed"() {name="input3"} : () -> tensor<?xf32>
  %bias1 = "pd.feed"() {name="input4"} : () -> tensor<?xf32>
  %bias2 = "pd.feed"() {name="input5"} : () -> tensor<?xf32>

  %d = "pd.elementwise_add"(%c, %bias) {axis=1:i32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e = "pd.relu6"(%d) {} : (tensor<?xf32>) -> tensor<?xf32>

  %c1 = "pd.matmul"(%e, %b1) {transpose_x=false, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %d1 = "pd.elementwise_add"(%c1, %bias1) {axis=1:i32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e1 = "pd.relu"(%d1) {} : (tensor<?xf32>) -> tensor<?xf32>

  %c2 = "pd.matmul"(%e1, %b2) {transpose_x=true, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %d2 = "pd.elementwise_add"(%c2, %bias2) {axis=1:i32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e2 = "pd.relu"(%d2) {} : (tensor<?xf32>) -> tensor<?xf32>
  
  "pd.fetch"(%e2) :(tensor<?xf32>)->()
}
