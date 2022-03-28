// RUN: infrtopt -phi-op-convert -core-op-fuse %s

// CHECK-LABEL: @ops
func @ops(%a:!core.lod_tensor<?xf32,0>, %b:!core.lod_tensor<?xf32,0>) {
  %g = "pd.elementwise_add"(%a, %b) {axis=1:si32} : (!core.lod_tensor<?xf32,0>, !core.lod_tensor<?xf32>) -> tensor<?xf32>
  %h = "pd.abs"(%g):(tensor<?xf32>) -> tensor<?xf32>
  core.return %h:tensor<?xf32>
}

// CHECK-LABEL: @op_execute
func @op_execute(%a:!core.lod_tensor<?xf32,0>, %b:!core.lod_tensor<?xf32,0>, %c:!core.lod_tensor<?xf32,0>)  -> !core.lod_tensor<?xf32,0> {
  %g = "pd.elementwise_add"(%a, %b) {axis=1:si32} : (!core.lod_tensor<?xf32,0>, !core.lod_tensor<?xf32>) -> tensor<?xf32>
  %h = "pd.abs"(%g):(tensor<?xf32>) -> tensor<?xf32>
  core.return %h:tensor<?xf32>
}
