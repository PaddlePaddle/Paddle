// RUN: infrtexec -i %s | FileCheck %s
// CHECK-LABEL: @basic_f32
func @basic_f32() -> f32 {
  %v0 = infrt.constant.f32 1.0
  %v1 = infrt.constant.f32 2.0
  %value = "infrt.add.f32"(%v0, %v1) : (f32, f32) -> f32

  // CHECK-NEXT: 3
  "infrt.print.f32"(%value) : (f32) -> ()

  infrt.return %value : f32
}

/// ================================================================
/// @caller call the other function @callee
func @callee.add.f32(%x : f32, %y : f32, %y1 : f32) -> f32 {
  %z = "infrt.add.f32"(%x, %y) : (f32, f32) -> f32
  %z1 = "infrt.add.f32"(%z, %y1) : (f32, f32) -> f32
  infrt.return %z1 : f32
}

// CHECK-LABEL: @caller.add.f32
func @caller.add.f32() -> f32 {
  %x = infrt.constant.f32 1.0
  %y = infrt.constant.f32 2.0
  %y1 = infrt.constant.f32 3.0
  %z = infrt.call @callee.add.f32(%x, %y, %y1) : (f32, f32, f32) -> f32

  // CHECK-NEXT: 6
  "infrt.print.f32"(%z) : (f32) -> ()
  infrt.return %z : f32
}
/// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

// CHECK-LABEL: @string_test
func @string_test() {
  %path = infrt.get_string("this is get_string op.")
  // CHECK-LABEL: string = this is get_string op.
  infrt.print_string(%path)
  infrt.return
}
