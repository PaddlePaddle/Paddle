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

#include <c10/util/intrusive_ptr.h>

#include <atomic>

#include "gtest/gtest.h"

// ============================================================================
// Helper: a minimal intrusive_ptr_target subclass that tracks destruction.
// ============================================================================

namespace {

class TestTarget : public c10::intrusive_ptr_target {
 public:
  explicit TestTarget(std::atomic<int>* destroy_count)
      : destroy_count_(destroy_count) {}

  ~TestTarget() override {
    destroy_count_->fetch_add(1, std::memory_order_relaxed);
  }

 private:
  std::atomic<int>* destroy_count_;
};

}  // namespace

// ============================================================================
// Weak reference lifecycle tests
// ============================================================================

// When a weak_intrusive_ptr outlives an intrusive_ptr, the object must NOT be
// deleted until the weak_intrusive_ptr is also destroyed.
TEST(IntrusivePtrLifecycleTest, WeakPtrKeepsObjectAliveAfterStrongReset) {
  std::atomic<int> destroy_count{0};

  c10::intrusive_ptr<TestTarget> strong =
      c10::make_intrusive<TestTarget>(&destroy_count);
  c10::weak_intrusive_ptr<TestTarget> weak(strong);

  // Object should be alive with both strong and weak references.
  ASSERT_EQ(destroy_count.load(), 0);
  ASSERT_EQ(strong.use_count(), 1u);

  // Destroy the strong reference.
  strong.reset();

  // The object must NOT have been deleted yet: the weak reference keeps it.
  ASSERT_EQ(destroy_count.load(), 0);

  // lock() should return an empty intrusive_ptr (strong count is 0).
  c10::intrusive_ptr<TestTarget> locked = weak.lock();
  ASSERT_FALSE(locked.defined());

  // expired() should be true.
  ASSERT_TRUE(weak.expired());

  // Destroying the weak reference should trigger deletion.
  weak.reset();
  ASSERT_EQ(destroy_count.load(), 1);
}

// When all strong references are released and no weak references exist,
// the object is deleted immediately.
TEST(IntrusivePtrLifecycleTest, ObjectDeletedImmediatelyWithNoWeakRefs) {
  std::atomic<int> destroy_count{0};

  {
    c10::intrusive_ptr<TestTarget> strong =
        c10::make_intrusive<TestTarget>(&destroy_count);
    // No weak references created.
    ASSERT_EQ(destroy_count.load(), 0);
  }  // strong goes out of scope here.

  // Without any weak references, the object should be deleted immediately.
  ASSERT_EQ(destroy_count.load(), 1);
}

// lock() must return empty when the strong reference count is zero, and
// must not resurrect the object.
TEST(IntrusivePtrLifecycleTest, LockReturnsEmptyAfterStrongGone) {
  std::atomic<int> destroy_count{0};

  c10::intrusive_ptr<TestTarget> strong =
      c10::make_intrusive<TestTarget>(&destroy_count);
  c10::weak_intrusive_ptr<TestTarget> weak(strong);

  strong.reset();  // Kill the strong reference.

  // lock() must return an empty (undefined) intrusive_ptr.
  c10::intrusive_ptr<TestTarget> result = weak.lock();
  ASSERT_FALSE(result.defined());
  ASSERT_EQ(result.get(), nullptr);

  // Object is still alive (weak reference holds it).
  ASSERT_EQ(destroy_count.load(), 0);
}

// Multiple weak references: object survives until the LAST weak ref is gone.
TEST(IntrusivePtrLifecycleTest, ObjectSurvivesUntilLastWeakRefGone) {
  std::atomic<int> destroy_count{0};

  c10::intrusive_ptr<TestTarget> strong =
      c10::make_intrusive<TestTarget>(&destroy_count);
  c10::weak_intrusive_ptr<TestTarget> weak1(strong);
  c10::weak_intrusive_ptr<TestTarget> weak2(strong);

  strong.reset();  // Strong gone; two weak refs remain.
  ASSERT_EQ(destroy_count.load(), 0);

  weak1.reset();  // First weak gone; one weak ref remains.
  ASSERT_EQ(destroy_count.load(), 0);

  weak2.reset();  // Last weak gone; object should be deleted now.
  ASSERT_EQ(destroy_count.load(), 1);
}

// Resetting multiple copies of a strong intrusive_ptr should not double-delete.
TEST(IntrusivePtrLifecycleTest, MultipleCopiesNoDoubleFree) {
  std::atomic<int> destroy_count{0};

  c10::intrusive_ptr<TestTarget> a =
      c10::make_intrusive<TestTarget>(&destroy_count);
  c10::intrusive_ptr<TestTarget> b = a;  // copy: refcount = 2
  c10::intrusive_ptr<TestTarget> c = a;  // copy: refcount = 3
  c10::weak_intrusive_ptr<TestTarget> weak(a);

  ASSERT_EQ(a.use_count(), 3u);

  a.reset();
  ASSERT_EQ(destroy_count.load(), 0);

  b.reset();
  ASSERT_EQ(destroy_count.load(), 0);

  c.reset();  // Strong count drops to zero; weak ref still alive.
  ASSERT_EQ(destroy_count.load(), 0);  // Not deleted yet.

  weak.reset();  // Last reference; object deleted.
  ASSERT_EQ(destroy_count.load(), 1);
}

// raw::intrusive_ptr::incref/decref should follow the same two-phase lifecycle.
TEST(IntrusivePtrLifecycleTest, RawIncrefDecrefTwoPhaseLifecycle) {
  std::atomic<int> destroy_count{0};

  // Create via make_intrusive (strong=1, weak=1).
  c10::intrusive_ptr<TestTarget> strong =
      c10::make_intrusive<TestTarget>(&destroy_count);
  TestTarget* raw = strong.get();

  // Create a weak reference.
  c10::weak_intrusive_ptr<TestTarget> weak(strong);

  // Manually add a strong reference via raw API, then decref via raw API.
  c10::raw::intrusive_ptr::incref(raw);  // strong = 2
  strong.reset();                        // strong = 1 via reset_()

  ASSERT_EQ(destroy_count.load(), 0);  // Still alive.

  // Decref the raw strong reference to zero.
  c10::raw::intrusive_ptr::decref(raw);  // strong = 0; implicit weak released.

  // Object is not yet deleted because weak reference still exists.
  ASSERT_EQ(destroy_count.load(), 0);

  weak.reset();  // Weak reference gone; object deleted.
  ASSERT_EQ(destroy_count.load(), 1);
}
