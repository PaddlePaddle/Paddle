// RUN: infrtopt --pd-op-fuse %s | FileCheck %s
// CHECK-LABEL: @main
func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2:tensor<?xf32>, %arg3:tensor<?xf32>, %arg4:tensor<?xf32>, %arg5:tensor<?xf32>, %arg6:tensor<?xf32>) -> tensor<?xf32> {

  // CHECK: %0 = "pd.FC"(%arg0, %arg1, %arg4) {in_num_col_dims = 1 : i32} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %c = "pd.matmul_v2"(%arg0, %arg1) {transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %d = "pd.elementwise_add"(%c, %arg4) {axis=1:si32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e = "pd.relu6"(%d) {} : (tensor<?xf32>) -> tensor<?xf32>

  // CHECK: %2 = "pd.FC"(%1, %arg2, %arg5) {in_num_col_dims = 1 : i32} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %c1 = "pd.matmul_v2"(%e, %arg2) {transpose_x=false, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %d1 = "pd.elementwise_add"(%c1, %arg5) {axis=1:si32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e1 = "pd.relu"(%d1) {} : (tensor<?xf32>) -> tensor<?xf32>

  // CHECK: %4 = "pd.FC"(%3, %arg3, %arg6) {in_num_col_dims = 1 : i32} : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %c2 = "pd.matmul_v2"(%e1, %arg3) {transpose_x=true, transpose_y=false} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %d2 = "pd.elementwise_add"(%c2, %arg6) {axis=1:si32} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %e2 = "pd.relu"(%d2) {} : (tensor<?xf32>) -> tensor<?xf32>
  infrt.return %e2:tensor<?xf32>
}
