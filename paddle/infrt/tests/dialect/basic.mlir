// RUN: infrtexec -i %s | FileCheck %s
// CHECK-LABEL: @basic_f32
func @basic_f32() -> f32 {
  %v0 = core.constant.f32 1.0
  %v1 = core.constant.f32 2.0
  %value = "core.add.f32"(%v0, %v1) : (f32, f32) -> f32

  // CHECK-NEXT: 3
  "core.print.f32"(%value) : (f32) -> ()

  core.return %value : f32
}

/// ================================================================
/// @caller call the other function @callee
func @callee.add.f32(%x : f32, %y : f32, %y1 : f32) -> f32 {
  %z = "core.add.f32"(%x, %y) : (f32, f32) -> f32
  %z1 = "core.add.f32"(%z, %y1) : (f32, f32) -> f32
  core.return %z1 : f32
}

// CHECK-LABEL: @caller.add.f32
func @caller.add.f32() -> f32 {
  %x = core.constant.f32 1.0
  %y = core.constant.f32 2.0
  %y1 = core.constant.f32 3.0
  %z = core.call @callee.add.f32(%x, %y, %y1) : (f32, f32, f32) -> f32

  // CHECK-NEXT: 6
  "core.print.f32"(%z) : (f32) -> ()
  core.return %z : f32
}
/// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
