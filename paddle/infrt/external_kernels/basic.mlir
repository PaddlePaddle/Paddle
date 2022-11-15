// CHECK: basic
func @basic() -> f32 {
  %v0 = infrt.constant.f32 1.0
  %v1 = infrt.constant.f32 2.0
  %v2 = "external.add.f32"(%v0, %v1) : (f32, f32) -> f32

  // CHECK: 1
  "external.print.f32"(%v0) : (f32) -> ()
  // CHECK: 2
  "external.print.f32"(%v1) : (f32) -> ()

  // CHECK: 3
  "external.print.f32"(%v2) : (f32) -> ()

  %v3 = "external.mul.f32"(%v2, %v1) : (f32, f32) -> f32

  // CHECK: 6
  "external.print.f32"(%v3) : (f32) -> ()

  infrt.return %v3 : f32
}
