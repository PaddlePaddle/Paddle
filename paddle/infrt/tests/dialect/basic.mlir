// RUN: infrtexec -i %s | FileCheck %s
// CHECK-LABEL: @basic_f32
func @basic_f32() -> f32 {
  %v0 = Infrt.constant.f32 1.0
  %v1 = Infrt.constant.f32 2.0
  %value = "Infrt.add.f32"(%v0, %v1) : (f32, f32) -> f32

  // CHECK-NEXT: 3
  "Infrt.print.f32"(%value) : (f32) -> ()

  Infrt.return %value : f32
}

/// ================================================================
/// @caller call the other function @callee
func @callee.add.f32(%x : f32, %y : f32, %y1 : f32) -> f32 {
  %z = "Infrt.add.f32"(%x, %y) : (f32, f32) -> f32
  %z1 = "Infrt.add.f32"(%z, %y1) : (f32, f32) -> f32
  Infrt.return %z1 : f32
}

// CHECK-LABEL: @caller.add.f32
func @caller.add.f32() -> f32 {
  %x = Infrt.constant.f32 1.0
  %y = Infrt.constant.f32 2.0
  %y1 = Infrt.constant.f32 3.0
  %z = Infrt.call @callee.add.f32(%x, %y, %y1) : (f32, f32, f32) -> f32

  // CHECK-NEXT: 6
  "Infrt.print.f32"(%z) : (f32) -> ()
  Infrt.return %z : f32
}
/// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
