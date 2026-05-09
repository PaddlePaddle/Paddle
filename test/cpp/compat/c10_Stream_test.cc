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

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Stream.h>
#include <c10/cuda/CUDAFunctions.h>

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAStream.h>
#endif

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "gtest/gtest.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/core/cuda_stream.h"
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
namespace {

using StreamCallbackGate = std::atomic<bool>;

#ifdef PADDLE_WITH_HIP
void BlockingStreamCallback(hipStream_t /*stream*/,
                            hipError_t /*status*/,
                            void* user_data) {
  auto* gate = static_cast<StreamCallbackGate*>(user_data);
  while (!gate->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void CreateRawStream(hipStream_t* stream) {
  C10_CUDA_CHECK(hipStreamCreate(stream));
}

void DestroyRawStream(hipStream_t stream) {
  C10_CUDA_CHECK(hipStreamDestroy(stream));
}

void ClearLastStreamError() { (void)hipGetLastError(); }
#else
void CUDART_CB BlockingStreamCallback(void* user_data) {
  auto* gate = static_cast<StreamCallbackGate*>(user_data);
  while (!gate->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void ClearLastStreamError() { (void)cudaGetLastError(); }
#endif

}  // namespace
#endif

// Test device_count() works in both CPU and CUDA builds
TEST(StreamTest, DeviceCount) {
  c10::DeviceIndex count = c10::cuda::device_count();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // In CUDA builds, should return actual device count (>= 0)
  EXPECT_GE(count, 0);
#else
  // In CPU-only builds, should return 0
  EXPECT_EQ(count, 0);
#endif
}

// ==================== native_handle ====================

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// CUDA stream: native_handle() should return the underlying cudaStream_t
// encoded as void*. For the default (null) stream the id is 0, so the
// pointer is nullptr; for a real stream it must be non-null.
TEST(StreamTest, NativeHandleCudaDefaultStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  c10::Stream s = c10::cuda::getDefaultCUDAStream().unwrap();
  // Default stream encodes nullptr (id == 0), so native_handle() == nullptr.
  EXPECT_EQ(s.native_handle(), nullptr);
}

TEST(StreamTest, NativeHandleCudaCurrentStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto cuda_stream = c10::cuda::getCurrentCUDAStream();
  c10::Stream s = cuda_stream.unwrap();
  // getCurrentCUDAStream wraps the real phi stream handle; calling
  // native_handle() must not throw.
  EXPECT_NO_THROW({ (void)s.native_handle(); });
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

// CPU stream: native_handle() is not supported and must throw.
TEST(StreamTest, NativeHandleCpuStreamThrows) {
  c10::Stream cpu_stream(c10::Stream::DEFAULT,
                         c10::Device(c10::DeviceType::CPU, 0));
  EXPECT_THROW({ (void)cpu_stream.native_handle(); }, std::exception);
}

// ==================== query ====================

// CPU stream is always ready.
TEST(StreamTest, QueryCpuStreamReturnsTrue) {
  c10::Stream cpu_stream(c10::Stream::DEFAULT,
                         c10::Device(c10::DeviceType::CPU, 0));
  EXPECT_TRUE(cpu_stream.query());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// A freshly-obtained CUDA stream with no pending work must report ready.
TEST(StreamTest, QueryCudaStreamReady) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto cuda_stream = c10::cuda::getCurrentCUDAStream();
  c10::Stream s = cuda_stream.unwrap();
  // synchronize first to ensure no pending work, then query should be true.
  EXPECT_NO_THROW(s.synchronize());
  EXPECT_TRUE(s.query());
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

// ==================== synchronize ====================

// CPU stream: synchronize() is a no-op and must not throw.
TEST(StreamTest, SynchronizeCpuStream) {
  c10::Stream cpu_stream(c10::Stream::DEFAULT,
                         c10::Device(c10::DeviceType::CPU, 0));
  EXPECT_NO_THROW(cpu_stream.synchronize());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// CUDA stream: synchronize() must complete without error.
TEST(StreamTest, SynchronizeCudaStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto cuda_stream = c10::cuda::getCurrentCUDAStream();
  c10::Stream s = cuda_stream.unwrap();
  EXPECT_NO_THROW(s.synchronize());
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

// ==================== getDefaultCUDAStream ====================

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// getDefaultCUDAStream must always return the null stream (id == 0),
// which corresponds to cudaStreamDefault on the device.
TEST(CUDAStreamTest, DefaultStreamIsNullStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto default_stream = c10::cuda::getDefaultCUDAStream();
  // id == 0 encodes cudaStreamDefault (the null stream, handle nullptr).
  EXPECT_EQ(default_stream.id(), static_cast<c10::StreamId>(0));
}

// getDefaultCUDAStream must be stable: calling it twice returns equal streams.
TEST(CUDAStreamTest, DefaultStreamIsStable) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto s1 = c10::cuda::getDefaultCUDAStream();
  auto s2 = c10::cuda::getDefaultCUDAStream();
  EXPECT_EQ(s1, s2);
}

TEST(CUDAStreamTest, GetStreamFromPoolBoolOverloadPreservesHighPriority) {
  if (!at::cuda::is_available()) {
    return;
  }
  auto low_priority_stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  auto high_priority_stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/true);
  auto explicit_high_priority_stream = c10::cuda::getStreamFromPool(-1);

  const int low_priority = low_priority_stream.priority();
  const int high_priority = high_priority_stream.priority();
  const int explicit_high_priority = explicit_high_priority_stream.priority();

  if (low_priority == explicit_high_priority) {
    return;
  }

  EXPECT_EQ(high_priority, explicit_high_priority);
  EXPECT_NE(high_priority, low_priority);
}

// After setCurrentCUDAStream redirects the current stream,
// getDefaultCUDAStream must still return the null stream.
TEST(CUDAStreamTest, DefaultStreamUnaffectedBySetCurrentCUDAStream) {
  if (!at::cuda::is_available()) {
    return;
  }
  // Snapshot the current stream before we touch it so we can
  // restore it afterward and avoid polluting subsequent tests.
  auto original_stream = c10::cuda::getCurrentCUDAStream();

  // Obtain a non-default stream from the pool.
  auto pool_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);

  // Redirect the current stream.
  c10::cuda::setCurrentCUDAStream(pool_stream);

  auto default_stream = c10::cuda::getDefaultCUDAStream();
  auto current_stream = c10::cuda::getCurrentCUDAStream();
  auto place = phi::GPUPlace(current_stream.device_index());

  // Default stream is still null; current stream has changed.
  EXPECT_EQ(default_stream.id(), static_cast<c10::StreamId>(0));
  EXPECT_NE(default_stream, current_stream);
  EXPECT_EQ(paddle::GetCurrentCUDAStream(place)->raw_stream(),
            current_stream.stream());

  // Restore the original current stream.
  c10::cuda::setCurrentCUDAStream(original_stream);
  EXPECT_EQ(paddle::GetCurrentCUDAStream(place)->raw_stream(),
            original_stream.stream());
}

// Verify getCurrentCUDAStream's thread-local semantics: a child thread
// that has not explicitly set a current stream sees the default stream,
// while each thread's own explicit set stays local to that thread.
TEST(CUDAStreamTest, SetCurrentCUDAStreamWriteIsolatedAcrossThreads) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto original_stream = c10::cuda::getCurrentCUDAStream();

  auto pool_a = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  auto pool_b = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);

  c10::cuda::setCurrentCUDAStream(pool_a);
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(), pool_a);

  std::thread unset_child([&]() {
    auto child_stream = c10::cuda::getCurrentCUDAStream(pool_a.device_index());
    EXPECT_EQ(child_stream,
              c10::cuda::getDefaultCUDAStream(pool_a.device_index()))
        << "A thread without c10 TLS must not inherit another thread's "
           "Paddle GPUContext stream.";
  });
  unset_child.join();

  std::thread t([&]() {
    c10::cuda::setCurrentCUDAStream(pool_b);
    EXPECT_EQ(c10::cuda::getCurrentCUDAStream(), pool_b);
  });
  t.join();

  // Main thread's thread-local is unaffected by the child's set —
  // getCurrentCUDAStream still hits pool_a from the main thread's TLS,
  // not pool_b that the child wrote to GPUContext.
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(), pool_a)
      << "Main thread's thread-local current stream should not be affected "
         "by another thread's setCurrentCUDAStream.";

  // Restore the original current stream.
  c10::cuda::setCurrentCUDAStream(original_stream);
}

TEST(CUDAStreamTest, ExplicitDefaultStreamDoesNotFallbackToGPUContext) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto original = c10::cuda::getCurrentCUDAStream();
  auto default_stream = c10::cuda::getDefaultCUDAStream();
  auto pool = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  c10::DeviceIndex device_index = pool.device_index();

  c10::cuda::setCurrentCUDAStream(default_stream);

  auto* ctx = static_cast<phi::GPUContext*>(
      paddle::experimental::DeviceContextPool::Instance().GetMutable(
          phi::GPUPlace(device_index)));
  phi::CUDAStream wrapper(phi::GPUPlace(device_index), pool.stream());
  ctx->SetCUDAStream(&wrapper, /*clear=*/false);

  auto cur = c10::cuda::getCurrentCUDAStream(device_index);
  EXPECT_EQ(cur, default_stream)
      << "An explicit c10 default stream must not be treated as unset TLS.";

  c10::cuda::setCurrentCUDAStream(original);
}

// Application-level pattern (the temp_modify reproducer in
// PaddleCppAPITest): even when the main thread blocks the c10 current
// stream via an event chain, a worker that uses its OWN independent
// non-blocking CUDA stream + CPU-side sync completes promptly and is
// not affected by the blocked stream.
//
// This also documents the application-level pattern needed for CUDA legacy
// default stream hazards: worker GPU work should use a worker-private stream.
TEST(CUDAStreamTest, IndependentWorkerStreamAvoidsBlockedCurrentStream) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto original = c10::cuda::getCurrentCUDAStream();
  auto pool = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  c10::cuda::setCurrentCUDAStream(pool);

#ifdef PADDLE_WITH_HIP
  hipStream_t enq_stream = nullptr;
  hipEvent_t event_start = nullptr;
  hipEvent_t event_end = nullptr;
  C10_CUDA_CHECK(hipStreamCreateWithFlags(&enq_stream, hipStreamNonBlocking));
  C10_CUDA_CHECK(hipEventCreateWithFlags(&event_start, hipEventDisableTiming));
  C10_CUDA_CHECK(hipEventCreateWithFlags(&event_end, hipEventDisableTiming));

  C10_CUDA_CHECK(hipEventRecord(event_start, pool.stream()));
  C10_CUDA_CHECK(hipStreamWaitEvent(enq_stream, event_start, 0));
#else
  cudaStream_t enq_stream = nullptr;
  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_end = nullptr;
  C10_CUDA_CHECK(cudaStreamCreateWithFlags(&enq_stream, cudaStreamNonBlocking));
  C10_CUDA_CHECK(
      cudaEventCreateWithFlags(&event_start, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&event_end, cudaEventDisableTiming));

  C10_CUDA_CHECK(cudaEventRecord(event_start, pool.stream()));
  C10_CUDA_CHECK(cudaStreamWaitEvent(enq_stream, event_start, 0));
#endif

  // Add a blocking callback on enq_stream (~200ms sleep), so pool_stream
  // (== c10 current stream) is effectively blocked on event_end.
  std::atomic<bool> callback_done{false};
#ifdef PADDLE_WITH_HIP
  C10_CUDA_CHECK(hipStreamAddCallback(
      enq_stream,
      [](hipStream_t, hipError_t, void* data) {
        auto* flag = static_cast<std::atomic<bool>*>(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        flag->store(true, std::memory_order_release);
      },
      &callback_done,
      0));
  C10_CUDA_CHECK(hipEventRecord(event_end, enq_stream));
  C10_CUDA_CHECK(hipStreamWaitEvent(pool.stream(), event_end, 0));
#else
  C10_CUDA_CHECK(cudaStreamAddCallback(
      enq_stream,
      [](cudaStream_t, cudaError_t, void* data) {
        auto* flag = static_cast<std::atomic<bool>*>(data);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        flag->store(true, std::memory_order_release);
      },
      &callback_done,
      0));
  C10_CUDA_CHECK(cudaEventRecord(event_end, enq_stream));
  C10_CUDA_CHECK(cudaStreamWaitEvent(pool.stream(), event_end, 0));
#endif

  // Worker thread uses its OWN non-blocking stream. It does NOT touch
  // c10's current stream (which is blocked). Sync should be immediate.
  std::packaged_task<void()> task([]() {
#ifdef PADDLE_WITH_HIP
    hipStream_t worker_stream = nullptr;
    C10_CUDA_CHECK(
        hipStreamCreateWithFlags(&worker_stream, hipStreamNonBlocking));
    C10_CUDA_CHECK(hipStreamSynchronize(worker_stream));
    C10_CUDA_CHECK(hipStreamDestroy(worker_stream));
#else
    cudaStream_t worker_stream = nullptr;
    C10_CUDA_CHECK(
        cudaStreamCreateWithFlags(&worker_stream, cudaStreamNonBlocking));
    C10_CUDA_CHECK(cudaStreamSynchronize(worker_stream));
    C10_CUDA_CHECK(cudaStreamDestroy(worker_stream));
#endif
  });
  auto future = task.get_future();
  std::thread worker(std::move(task));

  // Worker should complete promptly (well under 50ms) — its independent
  // stream has no dependency on enq_stream / event_end / pool_stream.
  auto status = future.wait_for(std::chrono::milliseconds(50));

  // Wait for callback to complete so pool_stream unblocks (with timeout
  // to prevent the test from hanging indefinitely).
  auto wait_start = std::chrono::steady_clock::now();
  while (!callback_done.load(std::memory_order_acquire)) {
    auto elapsed = std::chrono::steady_clock::now() - wait_start;
    if (elapsed > std::chrono::seconds(5)) {
      FAIL() << "Callback did not complete within 5s timeout";
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  future.wait();
  worker.join();

  EXPECT_EQ(status, std::future_status::ready)
      << "Worker with an independent non-blocking stream should complete "
         "promptly even when c10's current stream is blocked by an event "
         "chain on enq_stream.";

#ifdef PADDLE_WITH_HIP
  C10_CUDA_CHECK(hipEventDestroy(event_end));
  C10_CUDA_CHECK(hipEventDestroy(event_start));
  C10_CUDA_CHECK(hipStreamDestroy(enq_stream));
#else
  C10_CUDA_CHECK(cudaEventDestroy(event_end));
  C10_CUDA_CHECK(cudaEventDestroy(event_start));
  C10_CUDA_CHECK(cudaStreamDestroy(enq_stream));
#endif

  c10::cuda::setCurrentCUDAStream(original);
}

// Verify that a worker thread which has explicitly pinned its current
// stream (via setCurrentCUDAStream(pool_worker)) sees a stable result
// from getCurrentCUDAStream — the worker's thread-local "pinned"
// stream remains stable even while the main thread keeps switching its
// own current stream.
TEST(CUDAStreamTest, GetCurrentCUDAStreamStableForWorkerThatExplicitlySet) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto original = c10::cuda::getCurrentCUDAStream();
  auto pool_a = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  auto pool_b = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  auto pool_worker = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);

  // Main thread sets a non-default current stream first.
  c10::cuda::setCurrentCUDAStream(pool_a);

  std::atomic<bool> stop{false};
  std::atomic<int> samples_count{0};
  std::atomic<bool> stable{true};

  // Worker explicitly pins its TLS to pool_worker, then loops reading
  // its current stream. Main thread keeps switching its own TLS between
  // pool_a/pool_b in parallel.
  std::thread worker([&]() {
    c10::cuda::setCurrentCUDAStream(pool_worker);
    while (!stop.load(std::memory_order_acquire)) {
      auto s = c10::cuda::getCurrentCUDAStream();
      if (s != pool_worker) {
        stable.store(false, std::memory_order_release);
      }
      samples_count.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  });

  // Main thread keeps switching the current stream.
  for (int i = 0; i < 30; ++i) {
    c10::cuda::setCurrentCUDAStream((i % 2 == 0) ? pool_a : pool_b);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  stop.store(true, std::memory_order_release);
  worker.join();

  // Worker collected at least a few samples.
  EXPECT_GT(samples_count.load(std::memory_order_relaxed), 0);

  // Every sample equals pool_worker (worker's pinned stream is stable).
  EXPECT_TRUE(stable.load(std::memory_order_acquire))
      << "Worker thread's pinned current stream (pool_worker) should not be "
         "affected by main thread's setCurrentCUDAStream switches.";

  c10::cuda::setCurrentCUDAStream(original);
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
