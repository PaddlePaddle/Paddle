func @ops() {
  %a = pd.feed() {name="input0"} : tensor<?xf32>
  %b = pd.feed() {name="input1"}: tensor<?xf32>

  %c = "pd.matmul"(%a, %b) {transpose_x=true, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

  infrt.return
}
