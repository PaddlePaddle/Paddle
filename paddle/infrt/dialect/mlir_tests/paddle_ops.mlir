func @ops() {
  %a = pd.Feed() : tensor<?xf32>
  %b = pd.Feed() : tensor<?xf32>

  %c = "pd.Matmul"(%a, %b) {transpose_x=true, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

  infrt.return
}
