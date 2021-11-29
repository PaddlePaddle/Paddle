// CHECK-LABEL: @benchmark
func @benchmark() {
  // CHECK-LABEL: BM:add.f32:Count: 3
  // CHECK-LABEL: BM:add.f32:Duration(ns)
  // CHECK-LABEL: BM:add.f32:Time Min(ns)
  // CHECK-LABEL: BM:add.f32:Time 50%(ns)
  // CHECK-LABEL: BM:add.f32:Time 95%(ns)
  // CHECK-LABEL: BM:add.f32:Time 99%(ns)
  // CHECK-LABEL: BM:add.f32:CPU Min(ns)
  // CHECK-LABEL: BM:add.f32:CPU 50%(ns)
  // CHECK-LABEL: BM:add.f32:CPU 95%(ns)
  // CHECK-LABEL: BM:add.f32:CPU 99%(ns)
  // CHECK-LABEL: BM:add.f32:CPU utilization(percent)
  cinn.benchmark "add.f32"() duration_secs = 1, max_count = 3, num_warmup_runs = 3
  {
    %0 = cinn.constant.f32 1.0
    %1 = cinn.constant.f32 2.0
    %res = "cinn.add.f32"(%0, %1) : (f32, f32) -> f32
    "cinn.print.f32"(%res) : (f32) -> ()
    cinn.return %res : f32
  }
  cinn.return
}
