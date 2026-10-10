// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/memory/allocation/memory_history_recorder.h"

#include <atomic>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace paddle {
namespace memory {

namespace {

MemHistoryTraceEntry MakeEntry(MemHistoryAction action,
                               int device,
                               uintptr_t addr,
                               size_t size) {
  MemHistoryTraceEntry entry;
  entry.action = action;
  entry.device = device;
  entry.addr = addr;
  entry.size = size;
  entry.id = 0;
  entry.stream = 0;
  entry.time_us = 0;
  return entry;
}

}  // namespace

class MemoryHistoryRecorderTest : public ::testing::Test {
 protected:
  // The recorder is a process-wide singleton, so every test starts from a
  // known state and leaves recording off for the next one.
  void SetUp() override {
    MemoryHistoryRecorder::Instance().SetEnabled(false, 0);
    SetMemStackMinSize(0);
  }
  void TearDown() override {
    MemoryHistoryRecorder::Instance().SetEnabled(false, 0);
    SetMemStackMinSize(0);
  }
};

TEST_F(MemoryHistoryRecorderTest, DisabledByDefault) {
  EXPECT_FALSE(MemHistoryEnabled());
  // RecordMemHistory must be a no-op while disabled.
  RecordMemHistory(MemHistoryAction::kAlloc, 0, 0x1000, 128, 1, 0);
  EXPECT_TRUE(MemoryHistoryRecorder::Instance().GetTrace(0).empty());
}

TEST_F(MemoryHistoryRecorderTest, EnableThenRecordAndRead) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 16);
  EXPECT_TRUE(MemHistoryEnabled());

  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x1000, 256));
  rec.Record(MakeEntry(MemHistoryAction::kFreeRequested, 0, 0x1000, 256));
  rec.Record(MakeEntry(MemHistoryAction::kFreeCompleted, 0, 0x1000, 256));

  auto trace = rec.GetTrace(0);
  ASSERT_EQ(trace.size(), 3u);
  EXPECT_EQ(trace[0].action, MemHistoryAction::kAlloc);
  EXPECT_EQ(trace[1].action, MemHistoryAction::kFreeRequested);
  EXPECT_EQ(trace[2].action, MemHistoryAction::kFreeCompleted);
  EXPECT_EQ(trace[0].addr, 0x1000u);
  EXPECT_EQ(trace[0].size, 256u);
}

TEST_F(MemoryHistoryRecorderTest, RingWrapsAroundOldestFirst) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 4);
  for (uintptr_t i = 0; i < 6; ++i) {
    rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x100 + i, 8));
  }

  // Capacity is 4, so the two oldest events are overwritten and the result is
  // still returned oldest-first.
  auto trace = rec.GetTrace(0);
  ASSERT_EQ(trace.size(), 4u);
  EXPECT_EQ(trace[0].addr, 0x102u);
  EXPECT_EQ(trace[1].addr, 0x103u);
  EXPECT_EQ(trace[2].addr, 0x104u);
  EXPECT_EQ(trace[3].addr, 0x105u);
}

TEST_F(MemoryHistoryRecorderTest, ZeroCapacityRecordsNothing) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 0);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x1000, 128));
  EXPECT_TRUE(rec.GetTrace(0).empty());
}

TEST_F(MemoryHistoryRecorderTest, NegativeDeviceIsIgnored) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, -1, 0x1000, 128));
  EXPECT_TRUE(rec.GetTrace(-1).empty());
  EXPECT_TRUE(rec.GetTrace(0).empty());
}

TEST_F(MemoryHistoryRecorderTest, UnknownDeviceReturnsEmptyTrace) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x1000, 128));
  // Ring for device 7 was never created.
  EXPECT_TRUE(rec.GetTrace(7).empty());
}

TEST_F(MemoryHistoryRecorderTest, RingsArePerDevice) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0xA000, 32));
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 1, 0xB000, 64));
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 1, 0xB100, 64));

  ASSERT_EQ(rec.GetTrace(0).size(), 1u);
  ASSERT_EQ(rec.GetTrace(1).size(), 2u);
  EXPECT_EQ(rec.GetTrace(0)[0].addr, 0xA000u);
  EXPECT_EQ(rec.GetTrace(1)[1].size, 64u);
}

TEST_F(MemoryHistoryRecorderTest, SetEnabledClearsPreviousEvents) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x1000, 128));
  ASSERT_EQ(rec.GetTrace(0).size(), 1u);

  // Re-arming must drop everything recorded so far, otherwise a new session
  // would inherit a stale prefix.
  rec.SetEnabled(true, 8);
  EXPECT_TRUE(rec.GetTrace(0).empty());
}

TEST_F(MemoryHistoryRecorderTest, DisableClearsEventsAndFlag) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x1000, 128));
  rec.SetEnabled(false, 0);
  EXPECT_FALSE(MemHistoryEnabled());
  EXPECT_TRUE(rec.GetTrace(0).empty());
}

TEST_F(MemoryHistoryRecorderTest, ClearKeepsCapacityAndEnabledState) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x1000, 128));
  rec.Clear();
  EXPECT_TRUE(rec.GetTrace(0).empty());
  EXPECT_TRUE(MemHistoryEnabled());

  // Still usable after Clear(): capacity was preserved.
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x2000, 64));
  ASSERT_EQ(rec.GetTrace(0).size(), 1u);
  EXPECT_EQ(rec.GetTrace(0)[0].addr, 0x2000u);
}

TEST_F(MemoryHistoryRecorderTest, AnnotateMarksEveryExistingRing) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  // Annotate only reaches rings that already exist, so touch two devices.
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x1000, 128));
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 1, 0x2000, 128));

  rec.Annotate("gstep 7 begin");

  for (int dev : {0, 1}) {
    auto trace = rec.GetTrace(dev);
    ASSERT_EQ(trace.size(), 2u) << "device " << dev;
    EXPECT_EQ(trace[1].action, MemHistoryAction::kAnnotation);
    EXPECT_EQ(trace[1].op_name, "gstep 7 begin");
    EXPECT_EQ(trace[1].size, 0u);
    EXPECT_EQ(trace[1].addr, 0u);
    EXPECT_GT(trace[1].time_us, 0u);
  }
}

TEST_F(MemoryHistoryRecorderTest, AnnotateIsNoOpWhenDisabled) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x1000, 128));
  rec.SetEnabled(false, 8);
  rec.Annotate("ignored");
  EXPECT_TRUE(rec.GetTrace(0).empty());
}

// ---------------------------------------------------------------------------
// Per-thread op label / Python-stack-id guards
// ---------------------------------------------------------------------------

TEST_F(MemoryHistoryRecorderTest, LabelGuardIsNoOpWhenDisabled) {
  {
    MemLabelGuard guard("matmul");
    EXPECT_EQ(CurrentMemOpLabel(), nullptr);
  }
  EXPECT_EQ(CurrentMemOpLabel(), nullptr);
  EXPECT_TRUE(MemOpLabelStack().empty());
}

TEST_F(MemoryHistoryRecorderTest, LabelGuardNestsAndPops) {
  MemoryHistoryRecorder::Instance().SetEnabled(true, 8);
  EXPECT_EQ(CurrentMemOpLabel(), nullptr);
  {
    MemLabelGuard outer("fused_moe");
    ASSERT_NE(CurrentMemOpLabel(), nullptr);
    EXPECT_STREQ(CurrentMemOpLabel(), "fused_moe");
    {
      MemLabelGuard inner("matmul");
      EXPECT_STREQ(CurrentMemOpLabel(), "matmul");
    }
    // Innermost label restored after the nested scope exits.
    EXPECT_STREQ(CurrentMemOpLabel(), "fused_moe");
  }
  EXPECT_EQ(CurrentMemOpLabel(), nullptr);
}

TEST_F(MemoryHistoryRecorderTest, StackGuardIsNoOpWhenDisabled) {
  {
    MemStackGuard guard(false, 42);
    EXPECT_EQ(CurrentMemStackId(), 0u);
  }
  EXPECT_EQ(CurrentMemStackId(), 0u);
  EXPECT_TRUE(MemStackIdStack().empty());
}

TEST_F(MemoryHistoryRecorderTest, StackGuardNestsAndPops) {
  MemoryHistoryRecorder::Instance().SetEnabled(true, 8);
  EXPECT_EQ(CurrentMemStackId(), 0u);
  {
    MemStackGuard outer(true, 11);
    EXPECT_EQ(CurrentMemStackId(), 11u);
    {
      MemStackGuard inner(true, 22);
      EXPECT_EQ(CurrentMemStackId(), 22u);
    }
    // An outer wrapper (e.g. PyLayer.apply) stays active between nested ops.
    EXPECT_EQ(CurrentMemStackId(), 11u);
  }
  EXPECT_EQ(CurrentMemStackId(), 0u);
}

TEST_F(MemoryHistoryRecorderTest, GuardsStayBalancedWhenToggledMidScope) {
  MemoryHistoryRecorder::Instance().SetEnabled(true, 8);
  {
    MemLabelGuard label("op");
    MemStackGuard stack(true, 5);
    // Turning recording off inside the guarded scope must not unbalance the
    // pop in the destructor.
    MemoryHistoryRecorder::Instance().SetEnabled(false, 8);
  }
  EXPECT_TRUE(MemOpLabelStack().empty());
  EXPECT_TRUE(MemStackIdStack().empty());
}

// ---------------------------------------------------------------------------
// RecordMemHistory: label / stack-id attribution and the size threshold
// ---------------------------------------------------------------------------

TEST_F(MemoryHistoryRecorderTest, RecordMemHistoryPicksUpLabelAndStackId) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  {
    MemLabelGuard label("gaussian_random");
    MemStackGuard stack(true, 99);
    RecordMemHistory(MemHistoryAction::kAlloc, 0, 0x3000, 512, 7, 0);
  }
  auto trace = rec.GetTrace(0);
  ASSERT_EQ(trace.size(), 1u);
  EXPECT_EQ(trace[0].op_name, "gaussian_random");
  EXPECT_EQ(trace[0].stack_id, 99u);
  EXPECT_EQ(trace[0].id, 7u);
  EXPECT_GT(trace[0].time_us, 0u);
}

TEST_F(MemoryHistoryRecorderTest, StackIdOnlyAttachedToAllocEvents) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  {
    MemStackGuard stack(true, 77);
    RecordMemHistory(MemHistoryAction::kFreeCompleted, 0, 0x3000, 512, 0, 0);
  }
  auto trace = rec.GetTrace(0);
  ASSERT_EQ(trace.size(), 1u);
  EXPECT_EQ(trace[0].stack_id, 0u);
}

TEST_F(MemoryHistoryRecorderTest, StackMinSizeGatesStackAttribution) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  SetMemStackMinSize(1024);
  {
    MemStackGuard stack(true, 55);
    // Below the threshold: no stack recorded.
    RecordMemHistory(MemHistoryAction::kAlloc, 0, 0x1000, 512, 0, 0);
    // At/above the threshold: stack recorded.
    RecordMemHistory(MemHistoryAction::kAlloc, 0, 0x2000, 1024, 0, 0);
  }
  auto trace = rec.GetTrace(0);
  ASSERT_EQ(trace.size(), 2u);
  EXPECT_EQ(trace[0].stack_id, 0u);
  EXPECT_EQ(trace[1].stack_id, 55u);
}

TEST_F(MemoryHistoryRecorderTest, RecordMemHistoryWithoutGuardsIsUntagged) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  RecordMemHistory(MemHistoryAction::kAlloc, 0, 0x4000, 64, 0, 0);
  auto trace = rec.GetTrace(0);
  ASSERT_EQ(trace.size(), 1u);
  EXPECT_TRUE(trace[0].op_name.empty());
  EXPECT_EQ(trace[0].stack_id, 0u);
}

TEST_F(MemoryHistoryRecorderTest, StreamIsPreserved) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 8);
  RecordMemHistory(MemHistoryAction::kFreeRequested, 0, 0x5000, 64, 0, 0xDEAD);
  auto trace = rec.GetTrace(0);
  ASSERT_EQ(trace.size(), 1u);
  EXPECT_EQ(trace[0].stream, 0xDEADu);
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

TEST_F(MemoryHistoryRecorderTest, RecordAfterDisableIsDropped) {
  // Deterministic form of the disable race: a writer that already passed the
  // enabled check in RecordMemHistory() lands in Record() only after recording
  // was stopped. Python disables with its default (non-zero) max_entries, so a
  // capacity-only check would let the event through and events would reappear
  // after `_record_memory_history(enabled=None)` returned.
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 1024);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x1000, 128));
  ASSERT_EQ(rec.GetTrace(0).size(), 1u);

  rec.SetEnabled(false, 1u << 20);
  rec.Record(MakeEntry(MemHistoryAction::kAlloc, 0, 0x2000, 128));
  EXPECT_TRUE(rec.GetTrace(0).empty());
}

TEST_F(MemoryHistoryRecorderTest, ToggleWhileRecordingLeavesNoEventsAfterOff) {
  // Stress form of the same guarantee: once the last SetEnabled(false) returns
  // and the writers have drained, no event may remain. Probabilistic by nature,
  // and worth running under TSAN as well.
  auto& rec = MemoryHistoryRecorder::Instance();
  std::atomic<bool> stop{false};
  std::vector<std::thread> writers;
  for (int t = 0; t < 4; ++t) {
    writers.emplace_back([&stop, t] {
      for (uintptr_t i = 0; !stop.load(std::memory_order_relaxed); ++i) {
        // Mirrors the allocator hook sites: check the flag, then record.
        if (MemHistoryEnabled()) {
          RecordMemHistory(
              MemHistoryAction::kAlloc, t % 2, 0x1000 + i, 128, 0, 0);
        }
      }
    });
  }

  for (int i = 0; i < 500; ++i) {
    rec.SetEnabled(true, 1024);
    // Non-zero max_entries on the disable path, exactly like the Python API.
    rec.SetEnabled(false, 1u << 20);
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto& th : writers) th.join();

  EXPECT_FALSE(MemHistoryEnabled());
  for (int dev : {0, 1}) {
    EXPECT_TRUE(rec.GetTrace(dev).empty()) << "device " << dev;
  }
}

TEST_F(MemoryHistoryRecorderTest, LabelStacksAreThreadLocal) {
  auto& rec = MemoryHistoryRecorder::Instance();
  rec.SetEnabled(true, 64);
  MemLabelGuard outer("main_thread_op");
  MemStackGuard outer_stack(1);

  std::thread worker([] {
    // A different thread starts with empty stacks even though the main thread
    // is inside a guarded scope.
    EXPECT_EQ(CurrentMemOpLabel(), nullptr);
    EXPECT_EQ(CurrentMemStackId(), 0u);
    MemLabelGuard guard("worker_op");
    EXPECT_STREQ(CurrentMemOpLabel(), "worker_op");
  });
  worker.join();

  EXPECT_STREQ(CurrentMemOpLabel(), "main_thread_op");
  EXPECT_EQ(CurrentMemStackId(), 1u);
}

TEST_F(MemoryHistoryRecorderTest, ReconfigureConcurrentlyWithRecording) {
  // Regression guard: `capacity_` is written by SetEnabled() under rings_lock_
  // and read by Record() outside of it, so it must be atomic and the ring must
  // never exceed the requested bound. Run under TSAN to also catch the race.
  auto& rec = MemoryHistoryRecorder::Instance();
  constexpr size_t kCap = 64;
  rec.SetEnabled(true, kCap);

  std::atomic<bool> stop{false};
  std::vector<std::thread> recorders;
  for (int t = 0; t < 4; ++t) {
    recorders.emplace_back([&rec, &stop, t] {
      for (uintptr_t i = 0; !stop.load(std::memory_order_relaxed); ++i) {
        rec.Record(MakeEntry(MemHistoryAction::kAlloc, t % 2, 0x1000 + i, 128));
      }
    });
  }

  for (int i = 0; i < 200; ++i) {
    // Includes a zero capacity, which Record() must reject instead of
    // computing `% 0`.
    rec.SetEnabled(true, (i % 3 == 0) ? 8 : kCap);
    rec.SetEnabled(true, 0);
  }
  rec.SetEnabled(true, kCap);
  stop.store(true, std::memory_order_relaxed);
  for (auto& th : recorders) th.join();

  for (int dev : {0, 1}) {
    EXPECT_LE(rec.GetTrace(dev).size(), kCap) << "device " << dev;
  }
}

}  // namespace memory
}  // namespace paddle
