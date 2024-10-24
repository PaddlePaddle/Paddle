// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <atomic>
#include <cstdlib>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/framework/new_executor/workqueue/event_count.h"
#include "paddle/fluid/framework/new_executor/workqueue/run_queue.h"
#include "paddle/fluid/framework/new_executor/workqueue/thread_environment.h"
#include "paddle/phi/core/os_info.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

namespace paddle {
namespace framework {

template <typename Environment>
class ThreadPoolTempl {
 public:
  typedef typename Environment::Task Task;
  typedef RunQueue<Task, 1024> Queue;

  ThreadPoolTempl(const std::string& name,
                  int num_threads,
                  bool allow_spinning,
                  bool always_spinning,
                  Environment env = Environment())
      : env_(env),
        allow_spinning_(allow_spinning),
        always_spinning_(always_spinning),
        global_steal_partition_(EncodePartition(0, num_threads)),
        blocked_(0),
        done_(false),
        cancelled_(false),
        ec_(num_threads),
        num_threads_(num_threads),
        thread_data_(num_threads),
        name_(name) {
    // Calculate coprimes of all numbers [1, num_threads].
    // Coprimes are used for random walks over all threads in Steal
    // and NonEmptyQueueIndex. Iteration is based on the fact that if we take
    // a random starting thread index t and calculate num_threads - 1 subsequent
    // indices as (t + coprime) % num_threads, we will cover all threads without
    // repetitions (effectively getting a presudo-random permutation of thread
    // indices).
    assert(num_threads_ >= 1 && num_threads_ < kMaxThreads);
    all_coprimes_.reserve(num_threads_);
    for (int i = 1; i <= num_threads_; ++i) {
      all_coprimes_.emplace_back(i);
      ComputeCoprimes(i, &(all_coprimes_.back()));
    }
    for (int i = 0; i < num_threads_; i++) {
      SetStealPartition(i, EncodePartition(0, num_threads_));
      thread_data_[i].thread.reset(
          env_.CreateThread([this, i]() { WorkerLoop(i); }));
    }
  }

  ~ThreadPoolTempl() {
    done_ = true;

    // Now if all threads block without work, they will start exiting.
    // But note that threads can continue to work arbitrary long,
    // block, submit new work, unblock and otherwise live full life.
    if (!cancelled_) {
      ec_.Notify(true);
    } else {
      // Since we were cancelled, there might be entries in the queues.
      // Empty them to prevent their destructor from asserting.
      for (size_t i = 0; i < thread_data_.size(); i++) {
        thread_data_[i].queue.Flush();
      }
    }
    // Join threads explicitly (by destroying) to avoid destruction order within
    // this class.
    for (size_t i = 0; i < thread_data_.size(); ++i) {
      thread_data_[i].thread.reset();
    }
  }

  void SetStealPartitions(
      const std::vector<std::pair<unsigned, unsigned>>& partitions) {
    assert(partitions.size() == static_cast<std::size_t>(num_threads_));

    // Pass this information to each thread queue.
    for (int i = 0; i < num_threads_; i++) {
      const auto& pair = partitions[i];
      unsigned start = pair.first, end = pair.second;
      AssertBounds(start, end);
      unsigned val = EncodePartition(start, end);
      SetStealPartition(i, val);
    }
  }

  void AddTask(std::function<void()> fn) {
    AddTaskWithHint(std::move(fn), 0, num_threads_);
  }

  void AddTaskWithHint(std::function<void()> fn, int start, int limit) {
    Task t = env_.CreateTask(std::move(fn));
    PerThread* pt = GetPerThread();
    if (pt->pool == this) {
      // Worker thread of this pool, push onto the thread's queue.
      Queue& q = thread_data_[pt->thread_id].queue;
      t = q.PushFront(std::move(t));
    } else {
      // A free-standing thread (or worker of another pool), push onto a random
      // queue.
      assert(start < limit);
      assert(limit <= num_threads_);
      int num_queues = limit - start;
      int rnd = Rand(&pt->rand) % num_queues;
      assert(start + rnd < limit);
      Queue& q = thread_data_[start + rnd].queue;
      t = q.PushBack(std::move(t));
    }

    // Note: below we touch this after making w available to worker threads.
    // Strictly speaking, this can lead to a racy-use-after-free. Consider that
    // Schedule is called from a thread that is neither main thread nor a worker
    // thread of this pool. Then, execution of w directly or indirectly
    // completes overall computations, which in turn leads to destruction of
    // this. We expect that such scenario is prevented by program, that is,
    // this is kept alive while any threads can potentially be in Schedule.
    if (!t.f) {
      // Allow 'false positive' which makes a redundant notification.
      VLOG(6) << "Add task, Notify";
      ec_.Notify(false);
    } else {
      env_.ExecuteTask(t);  // Push failed, execute directly.
    }
  }

  void Cancel() {
    cancelled_ = true;
    done_ = true;

    // Wake up the threads without work to let them exit on their own.
    ec_.Notify(true);
  }

  void WaitThreadsExit() {
    for (size_t i = 0; i < thread_data_.size(); ++i) {
      thread_data_[i].thread->WaitExit();
    }
  }

  size_t NumThreads() const { return num_threads_; }

  int CurrentThreadId() const {
    const PerThread* pt = const_cast<ThreadPoolTempl*>(this)->GetPerThread();
    if (pt->pool == this) {
      return pt->thread_id;
    } else {
      return -1;
    }
  }

 private:
  // Create a single atomic<int> that encodes start and limit information for
  // each thread.
  // We expect num_threads_ < 65536, so we can store them in a single
  // std::atomic<unsigned>.
  // Exposed publicly as static functions so that external callers can reuse
  // this encode/decode logic for maintaining their own thread-safe copies of
  // scheduling and steal domain(s).
  static const int kMaxPartitionBits = 16;
  static const int kMaxThreads = 1 << kMaxPartitionBits;

  inline unsigned EncodePartition(unsigned start, unsigned limit) {
    return (start << kMaxPartitionBits) | limit;
  }

  inline void DecodePartition(unsigned val, unsigned* start, unsigned* limit) {
    *limit = val & (kMaxThreads - 1);
    val >>= kMaxPartitionBits;
    *start = val;
  }

  void AssertBounds(int start, int end) {
    assert(start >= 0);
    assert(start < end);  // non-zero sized partition
    assert(end <= num_threads_);
  }

  inline void SetStealPartition(size_t i, unsigned val) {
    thread_data_[i].steal_partition.store(val, std::memory_order_relaxed);
  }

  inline unsigned GetStealPartition(int i) {
    return thread_data_[i].steal_partition.load(std::memory_order_relaxed);
  }

  inline void ComputeCoprimes(int n, std::vector<unsigned>* coprimes) {
    for (int i = 1; i <= n; i++) {
      unsigned a = i;
      unsigned b = n;
      // If GCD(a, b) == 1, then a and b are coprimes.
      while (b != 0) {
        unsigned tmp = a;
        a = b;
        b = tmp % b;
      }
      if (a == 1) {
        coprimes->push_back(i);
      }
    }
  }

  typedef typename Environment::EnvThread Thread;

  struct PerThread {
    constexpr PerThread() : pool(NULL), rand(0), thread_id(-1) {}
    ThreadPoolTempl* pool;  // Parent pool, or null for normal threads.
    uint64_t rand;          // Random generator state.
    int thread_id;          // Worker thread index in pool.
  };

  struct ThreadData {
    constexpr ThreadData() : thread(), steal_partition(0), queue() {}
    std::unique_ptr<Thread> thread;
    std::atomic<unsigned> steal_partition;
    Queue queue;
  };

  Environment env_;
  const bool allow_spinning_;
  const bool always_spinning_;
  std::vector<std::vector<unsigned>> all_coprimes_;
  unsigned global_steal_partition_;
  std::atomic<unsigned> blocked_;
  std::atomic<bool> done_;
  std::atomic<bool> cancelled_;
  EventCount ec_;
  const int num_threads_;
  std::vector<ThreadData> thread_data_;
  std::string name_;

  // Main worker thread loop.
  void WorkerLoop(int thread_id) {
    std::string thr_name = name_ + "_thread_" + std::to_string(thread_id);
    VLOG(1) << thr_name << " started ";
    phi::SetCurrentThreadName(thr_name);
    PerThread* pt = GetPerThread();
    pt->pool = this;
    pt->rand = GlobalThreadIdHash();
    pt->thread_id = thread_id;
    Queue& q = thread_data_[thread_id].queue;
    EventCount::Waiter* waiter = ec_.GetWaiter(thread_id);
    // TODO(dvyukov,rmlarsen): The time spent in NonEmptyQueueIndex() is
    // proportional to num_threads_ and we assume that new work is scheduled at
    // a constant rate, so we set spin_count to 5000 / num_threads_. The
    // constant was picked based on a fair dice roll, tune it.
    const int spin_count =
        allow_spinning_ && num_threads_ > 0 ? 5000 / num_threads_ : 0;
    if (num_threads_ == 1) {
      // For num_threads_ == 1 there is no point in going through the expensive
      // steal loop. Moreover, since NonEmptyQueueIndex() calls PopBack() on the
      // victim queues it might reverse the order in which ops are executed
      // compared to the order in which they are added, which tends to be
      // counter-productive for the types of I/O workloads the single thread
      // pools tend to be used for.
      while (!cancelled_) {
        Task t = q.PopFront();
        for (int i = 0; i < spin_count && !t.f; i++) {
          if (!cancelled_.load(std::memory_order_relaxed)) {
            t = q.PopFront();
          }
        }
        if (!t.f) {
          if (!WaitForWork(waiter, &t)) {
            return;
          }
        }
        if (t.f) {
          env_.ExecuteTask(t);
        }
      }
    } else {
      while (!cancelled_) {
        Task t = q.PopFront();
        if (!t.f) {
          t = LocalSteal();
          if (!t.f) {
            t = GlobalSteal();
            if (!t.f) {
              if (allow_spinning_) {
                for (int i = 0; i < spin_count && !t.f; i++) {
                  if (!cancelled_.load(std::memory_order_relaxed)) {
                    t = GlobalSteal();
                  } else {
                    return;
                  }
                }
              }
              if (!t.f) {
                if (!WaitForWork(waiter, &t)) {
                  return;
                }
              }
            }
          }
        }
        if (t.f) {
          env_.ExecuteTask(t);
        }
      }
    }
  }

  // Steal tries to steal work from other worker threads in the range [start,
  // limit) in best-effort manner.
  Task Steal(unsigned start, unsigned limit) {
    PerThread* pt = GetPerThread();
    const size_t size = limit - start;
    unsigned r = Rand(&pt->rand);
    // Reduce r into [0, size) range, this utilizes trick from
    // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    assert(all_coprimes_[size - 1].size() < (1 << 30));
    unsigned victim = ((uint64_t)r * (uint64_t)size) >> 32;
    unsigned index =
        ((uint64_t)all_coprimes_[size - 1].size() * (uint64_t)r) >> 32;
    unsigned inc = all_coprimes_[size - 1][index];

    for (unsigned i = 0; i < size; i++) {
      assert(start + victim < limit);
      Task t = thread_data_[start + victim].queue.PopBack();
      if (t.f) {
        return t;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return Task();
  }

  // Steals work within threads belonging to the partition.
  Task LocalSteal() {
    PerThread* pt = GetPerThread();
    unsigned partition = GetStealPartition(pt->thread_id);
    // If thread steal partition is the same as global partition, there is no
    // need to go through the steal loop twice.
    if (global_steal_partition_ == partition) return Task();
    unsigned start, limit;
    DecodePartition(partition, &start, &limit);
    AssertBounds(start, limit);

    return Steal(start, limit);
  }

  // Steals work from any other thread in the pool.
  Task GlobalSteal() { return Steal(0, num_threads_); }

  // WaitForWork blocks until new work is available (returns true), or if it is
  // time to exit (returns false). Can optionally return a task to execute in t
  // (in such case t.f != nullptr on return).
  bool WaitForWork(EventCount::Waiter* waiter, Task* t) {
    assert(t != nullptr && !t->f);
    // We already did best-effort emptiness check in Steal, so prepare for
    // blocking.
    ec_.Prewait();
    if (cancelled_) {
      ec_.CancelWait();
      return false;
    }

    // Number of blocked threads is used as notification condition.
    // We must increase the counter before the emptiness check.
    blocked_++;

    // Now do a reliable emptiness check.
    int victim = NonEmptyQueueIndex();
    if (victim != -1) {
      ec_.CancelWait();
      *t = thread_data_[victim].queue.PopBack();
      blocked_--;
      return true;
    }

    // Number of blocked threads is used as termination condition.
    // If we are shutting down and all worker threads blocked without work,
    // that's we are done.
    if (done_ && blocked_ == static_cast<unsigned>(num_threads_)) {
      ec_.CancelWait();
      // Almost done, but need to re-check queues.
      // Consider that all queues are empty and all worker threads are preempted
      // right after incrementing blocked_ above. Now a free-standing thread
      // submits work and calls destructor (which sets done_). If we don't
      // re-check queues, we will exit leaving the work unexecuted.
      if (NonEmptyQueueIndex() != -1) {
        // Note: we must not pop from queues before we decrement blocked_,
        // otherwise the following scenario is possible. Consider that instead
        // of checking for emptiness we popped the only element from queues.
        // Now other worker threads can start exiting, which is bad if the
        // work item submits other work. So we just check emptiness here,
        // which ensures that all worker threads exit at the same time.
        blocked_--;
        return true;
      }
      // Reached stable termination state.
      ec_.Notify(true);
      return false;
    }

    // Cancel wait if always_spinning_
    if (always_spinning_) {
      ec_.CancelWait();
      blocked_--;
      return true;
    }

    // Wait for work
    phi::RecordEvent record(
        "WaitForWork", phi::TracerEventType::UserDefined, 10);
    ec_.CommitWait(waiter);
    blocked_--;
    return true;
  }

  int NonEmptyQueueIndex() {
    PerThread* pt = GetPerThread();
    // We intentionally design NonEmptyQueueIndex to steal work from
    // anywhere in the queue so threads don't block in WaitForWork() forever
    // when all threads in their partition go to sleep. Steal is still local.
    const size_t size = thread_data_.size();
    unsigned r = Rand(&pt->rand);
    unsigned inc = all_coprimes_[size - 1][r % all_coprimes_[size - 1].size()];
    unsigned victim = r % size;
    for (unsigned i = 0; i < size; i++) {
      if (!thread_data_[victim].queue.Empty()) {
        return victim;
      }
      victim += inc;
      if (victim >= size) {
        victim -= size;
      }
    }
    return -1;
  }

  static inline uint64_t GlobalThreadIdHash() {
    return std::hash<std::thread::id>()(std::this_thread::get_id());
  }

  inline PerThread* GetPerThread() {
    static thread_local PerThread per_thread_;
    PerThread* pt = &per_thread_;
    return pt;
  }

  static inline unsigned Rand(uint64_t* state) {
    uint64_t current = *state;
    // Update the internal state
    *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
    // Generate the random output (using the PCG-XSH-RS scheme)
    return static_cast<unsigned>((current ^ (current >> 22)) >>
                                 (22 + (current >> 61)));
  }
};

using NonblockingThreadPool = ThreadPoolTempl<StlThreadEnvironment>;

}  // namespace framework
}  // namespace paddle
