// RUN: infrtopt -phi-op-convert -infrt-op-fuse %s

// CHECK-LABEL: @ops
func @ops() {
  %a = pd.feed() {name="input0"} : !infrt.lod_tensor<?xf32,0>
  %b = pd.feed() {name="input1"} : !infrt.lod_tensor<?xf32,0>
  %d = pd.feed() {name="input3"} : !infrt.lod_tensor<3x4x9xf32, 0>
  %g = "pd.elementwise_add"(%a, %b) {axis=1:si32} : (!infrt.lod_tensor<?xf32,0>, !infrt.lod_tensor<?xf32>) -> tensor<?xf32>
  %h = "pd.abs"(%g):(tensor<?xf32>) -> tensor<?xf32>
  "pd.fetch"(%h) {name="output"} :(tensor<?xf32>)->()
}

// CHECK-LABEL: @op_execute
func @op_execute(%a:!infrt.lod_tensor<?xf32,0>, %b:!infrt.lod_tensor<?xf32,0>, %c:!infrt.lod_tensor<?xf32,0>)  -> !infrt.lod_tensor<?xf32,0> {
  %g = "pd.elementwise_add"(%a, %b) {axis=1:si32} : (!infrt.lod_tensor<?xf32,0>, !infrt.lod_tensor<?xf32>) -> tensor<?xf32>
  %h = "pd.abs"(%g):(tensor<?xf32>) -> tensor<?xf32>
  "pd.fetch"(%h) {name="output"} :(tensor<?xf32>)->()
}
