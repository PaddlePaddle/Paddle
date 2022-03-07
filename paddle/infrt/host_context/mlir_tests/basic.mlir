// CHECK-LABEL: basic
func @basic() -> f32 {
  %v0 = Infrt.constant.f32 1.0
  %v1 = Infrt.constant.f32 2.0
  %v2 = "Infrt.add.f32"(%v0, %v1) : (f32, f32) -> f32

  // CHECK: 1
  "Infrt.print.f32"(%v0) : (f32) -> ()
  // CHECK: 2
  "Infrt.print.f32"(%v1) : (f32) -> ()

  // CHECK: 3
  "Infrt.print.f32"(%v2) : (f32) -> ()

  %v3 = "Infrt.mul.f32"(%v2, %v1) : (f32, f32) -> f32

  // CHECK: 6
  "Infrt.print.f32"(%v3) : (f32) -> ()

  Infrt.return %v3 : f32
}

// CHECK-LABEL: basic1
// Check the mlir executor can work with more than one function in a file.
func @basic1() -> () {
  %v0 = Infrt.constant.f32 1.0
  "Infrt.print.f32"(%v0) : (f32) -> ()
  // CHECK: 1
  Infrt.return
}