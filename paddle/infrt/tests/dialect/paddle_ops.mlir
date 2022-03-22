// RUN: infrtopt %s | FileCheck %s
// CHECK-LABEL: @ops
func @ops() {
  %a = pd.feed() {name="input0"} : tensor<?xf32>
  %b = pd.feed() {name="input1"}: tensor<?xf32>
  %d = pd.feed() {name="input3"}: !infrt.lod_tensor<3x4x9xf32, 0>
  %c = "pd.matmul"(%a, %b) {transpose_x=true, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  infrt.return
}
