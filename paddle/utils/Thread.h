/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <thread>
#include "Logging.h"
#include "Util.h"

#include "Queue.h"
#include "ThreadLocal.h"

#include <future>

namespace paddle {

/**
 * A simple wrapper for std::thread
 */

class Thread {
public:
  /**
   * @brief Construct Function. Default thread pointer is null.
   */
  Thread() { thread_ = nullptr; }

  virtual ~Thread() {}

  /**
   * @brief Creat a new thread and call *run()* function.
   */
  void start() {
    thread_.reset(new std::thread([this]() { this->run(); }));
  }

  /**
   * @brief Detach the thread.
   * It don't need to be waited until it finish.
   */
  void detach() { thread_->detach(); }

  /**
   * @brief Join the thread.
   * It should be waited until it finish.
   */
  void join() { thread_->join(); }

  /**
   * @brief Define what to be done on this thread through override this
   * function.
   */
  virtual void run() = 0;

protected:
  std::unique_ptr<std::thread> thread_;
};

/**
 * ThreadWorker maintains a job queue. It executes the jobs in the job queue
 * sequentianlly in a separate thread.
 *
 * Use addJob() to add a new job to the job queue.
 */
class ThreadWorker : protected Thread {
public:
  typedef std::function<void()> JobFunc;

  /**
   * @brief Construct Function. Default size of job queue is 0 and not stopping.
   */
  ThreadWorker() : stopping_(false), empty_(true) { start(); }

  /**
   * @brief Destruct Function.
   * If it's running, wait until all job finish and then stop it.
   */
  ~ThreadWorker() {
    if (!stopping_) {
      wait();
      stop();
    }
  }

  /**
   * @brief Finish current running job and quit the thread.
   */
  void stop() {
    stopping_ = true;
    jobs_.enqueue([]() {});
    join();
  }

  /**
   * @brief Add a new job to the job queue.
   */
  void addJob(JobFunc func) {
    empty_ = false;
    jobs_.enqueue(func);
  }

  /**
   * @brief Wait until all jobs was done (the job queue was empty).
   */
  void wait() {
    finishCV_.wait([this] { return empty_; });
  }

protected:
  /**
   * @brief Execute jobs in the job queue sequentianlly,
   * @note If finish all the jobs in the job queue,
   * notifies all the waiting threads the job queue was empty.
   */
  virtual void run() {
    while (true) {
      JobFunc func = jobs_.dequeue();
      if (stopping_) break;
      func();
      if (jobs_.empty()) {
        finishCV_.notify_all([this] { empty_ = true; });
      }
    }
  }

  Queue<JobFunc> jobs_;
  bool stopping_;
  LockedCondition finishCV_;
  bool empty_;
};

/**
 * SyncThreadPool maintains a pool of threads.
 * It executes the job use all workers in the pool.
 *
 * Use exec() to run a new job, job complete when exec returned.
 * Only one job can exec simultaneously.
 *
 * Each worker has an tid whose range is [0, getNumThreads()).
 * JobFunc can use tid to divide input data.
 */
class SyncThreadPool {
public:
  typedef std::function<void(int tid, size_t numThreads)> JobFunc;

  /**
   * @brief Construct Function. No thread will be created.
   */
  SyncThreadPool() : jobStartBarrier_(0), jobFinishBarrier_(0) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @brief Construct Fucntion. Create numWorkers of threads in the pool.
   * @param[in] numWorkers Number of the workers in the pool.
   * @param[in] checkOwner Default true. If checkOwner is true,
   * this sync thread pool should be used by it's owner thread.
   */
  explicit SyncThreadPool(size_t numWorkers, bool checkOwner = true)
      : stopping_(false),
        jobStartBarrier_(numWorkers + 1),
        jobFinishBarrier_(numWorkers + 1),
        jobFunc_(nullptr),
        checkOwner_(checkOwner) {
    ownerThreadId_ = getTID();
    workers_.resize(numWorkers);
    start();
  }

  ~SyncThreadPool() {
    if (!stopping_) {
      stop();
    }
  }

  /**
   * @brief Return num of threads in the pool.
   */
  size_t getNumThreads() { return workers_.size(); }

  /**
   * @brief Execute a job using all the theads in the pool.
   * @param[in] jobFunc The function to be executed.
   * @param[in] ownerFunc Owner thread can do something in owerFunc when job
   * executing.
   * @note For the ownerFunc, tid=getNumThreads().
   */
  void exec(JobFunc jobFunc, JobFunc ownerFunc = nullptr) {
    if (checkOwner_) {
      CHECK_EQ(ownerThreadId_, getTID())
          << "this sync thread pool should be used in one thread";
    }

    CHECK(jobFunc_ == nullptr);
    jobFunc_ = jobFunc;
    jobStartBarrier_.wait();  // notify worker thread start job

    if (ownerFunc) {
      ownerFunc(workers_.size(), workers_.size());
    }

    jobFinishBarrier_.wait();  // wait all worker thread complete
    jobFunc_ = nullptr;
  }

  /**
   * @brief Execute a job using all the threads in the pool.
   * And the owner thread will do the same job.
   * @param jobFunc The job to be executed.
   * @note  Assume that JobFunc will execute numThread + 1 times,
   * with tid ranging [0,numThread]. The thread whose tid is numThread
   * is the owner thread.
   */
  void execPlusOwner(JobFunc jobFunc) { exec(jobFunc, jobFunc); }

  /**
   * @brief Execute a job if has pool, else use caller thread as a worker.
   * @param[in] pool The pool to execute the job.
   * @param[in] jobFunc The job to be excuted.
   */
  static void execHelper(SyncThreadPool* pool, JobFunc jobFunc) {
    if (pool) {
      pool->exec(jobFunc);
    } else {
      jobFunc(0, 1);
    }
  }

protected:
  /**
   * @brief Start all the workers in the pool, call their run() function.
   */
  void start() {
    for (size_t i = 0; i < workers_.size(); ++i) {
      workers_[i].reset(
          new std::thread([this](int tid) { this->run(tid); }, i));
    }
  }

  /**
   * @brief Stop all the workers in the pool.
   */
  void stop() {
    stopping_ = true;
    // notify worker thread to stop
    jobStartBarrier_.wait();

    // stop workers
    for (auto& thread : workers_) {
      if (thread) {
        thread->join();
        thread.reset(nullptr);
      }
    }
  }

  /**
   * @brief Execute the jobFunc_ using the worker thread tid, if not stopping.
   */
  void run(int tid) {
    VLOG(1) << "SyncThreadPool worker thread " << tid;
    // init seed deterministic, but differs from global srand()
    ThreadLocalRand::initThreadSeed(tid + workers_.size());

    while (true) {
      jobStartBarrier_.wait();  // wait job

      if (stopping_) {
        break;
      }

      jobFunc_(tid, workers_.size());

      jobFinishBarrier_.wait();  // notify job complete
    }
  }

protected:
  pid_t ownerThreadId_;
  bool stopping_;
  ThreadBarrier jobStartBarrier_;
  ThreadBarrier jobFinishBarrier_;

  JobFunc jobFunc_;
  bool checkOwner_;
  std::vector<std::unique_ptr<std::thread>> workers_;
};

/**
 * MultiThreadWorker maintains a job queue and a result queue.
 * It executes the jobs in the job queue and puts the results into the
 * result queue sequentially in multi separate threads.
 *
 * Add jobs:
 *
 *    Use addJob() to add a new job to the job queue
 *        (the user added jobs should not return nullptr).
 *
 *    Use stopAddJob() to stop adding new jobs to the job queue
 *        (addJob() can not be called after stopAddJob()).
 *
 * Normal stop:
 *
 *    Use waitResult() to get the results until nullptr is returned.
 *    Use stop() to exit normally
 *        (stopAddJob() should be called first).
 *
 * Force stop:
 *
 *    Use forceStop() to exit forcibly even though there are remaining jobs in
 * the
 * job queue.
 */
template <class T>
class MultiThreadWorker {
public:
  typedef T ResultType;
  typedef std::shared_ptr<ResultType> ResultPtrType;
  typedef std::function<ResultPtrType()> JobFunc;
  /**
   * @brief Construct Function. Initialize the multithread worker.
   * @param[in] workerNum Number of the workers.
   * @param[in] queueCapacity Capapcity of the result queue.
   */
  MultiThreadWorker(size_t workerNum, size_t queueCapacity)
      : stopping_(false),
        jobAdding_(true),
        nullResultNum_(0),
        results_(queueCapacity) {
    workers_.resize(workerNum);
    for (auto& worker : workers_) {
      worker.reset(new std::thread([this]() { this->run(); }));
    }
  }

  /**
   * @brief Destruct Function. Force stop the workers
   * even though there are remaining jobs in the job queue.
   */
  virtual ~MultiThreadWorker() { forceStop(); }

  /**
   * @brief Stop all the workers normally.
   * @note stopAddJob() should be called before it.
   */
  void stop() {
    CHECK(!jobAdding_) << "stopAddJob() should be called before stop()";
    for (auto& worker : workers_) {
      if (worker) {
        worker->join();
        worker = nullptr;
      }
    }
    stopping_ = true;
  }

  /**
   * @brief Stop all the workers forcibly.
   * @note This function will call stopAddJob() first
   * and empty the result queue.
   */
  void forceStop() {
    if (!stopping_) {
      stopping_ = true;
      stopAddJob();
      while (nullptr != waitResult()) {
      }
      stop();
    }
  }

  /**
   * @brief Add a job to the job queue.
   * @note Job can not be added after calling stopAddJob().
   */
  void addJob(JobFunc func) {
    CHECK(jobAdding_) << "addJob() can not be called after stopAddJob()";
    jobs_.enqueue(func);
  }

  /**
   * @brief Stop adding new jobs to the job queue.
   * @note This fuction enqueue a return nullptr function to the job queue.
   */
  void stopAddJob() {
    for (size_t i = 0; i < workers_.size(); ++i) {
      jobs_.enqueue([]() { return nullptr; });
    }
    jobAdding_ = false;
  }

  /**
   * @brief Dequeue the first result in the result queue and return it.
   * @note If the result queue is empty, wait until it's not empty
   * or return nullptr if all the results have been returned.
   */
  ResultPtrType waitResult() {
    while (true) {
      ResultPtrType result = results_.dequeue();
      if (result) {
        return result;
      }

      ++nullResultNum_;
      if (nullResultNum_ == workers_.size()) {
        return nullptr;
      }
    }
  }

  /**
   * @brief The result queue is empty or not.
   * @return true if empty.
   */
  bool testResult() { return results_.empty(); }

protected:
  /**
   * @brief Do the jobs in the job queue sequentianlly
   * and enqueue the result into the result queue.
   * @note A nullptr will be enqueued into the resulte queue, when a worker
   * finished.
   */
  virtual void run() {
    while (true) {
      JobFunc func = jobs_.dequeue();
      ResultPtrType result = func();
      if (result == nullptr || stopping_) {
        // When a worker finished, a nullptr would be enqueued into results_
        results_.enqueue(nullptr);
        break;
      }
      results_.enqueue(result);
    }
  }

  bool stopping_;
  bool jobAdding_;
  size_t nullResultNum_;
  Queue<JobFunc> jobs_;
  BlockingQueue<ResultPtrType> results_;
  std::vector<std::unique_ptr<std::thread>> workers_;
};

/**
 * AsyncThreadPool maintains a job queue and threads pool.
 * It executes the jobs from queue asynchronously.
 *
 * Add jobs:
 *
 *    Use addJob() to add a new job to the job queue and get a std::future
 *    result. The caller's thread continues running. Call std::future::get()
 *    when the result's value is needed, and the caller's thread may be
 *    blocked until thread-pool finished the job.
 *
 *    Use addBatchJobs() to add a batch of jobs.
 *    Unlike addJob()'s asynchronization, addBatchJobs will block caller's
 *    thread until all jobs in the batch are finished.
 *
 * Stop:
 *    Use stop() to stop the thread pool. Job can be added once stopped.
 *
 * Process-wide Singleton:
 *    Use AsyncThreadPool::ProcessChannel(N) first to create N threads.
 *    Then call AsyncThreadPool::ProcessChannel() to get the process-wide global
 *    thread pool.
 */
class AsyncThreadPool {
public:
  typedef std::function<void()> JobFunc;

  AsyncThreadPool() { LOG(FATAL) << "Not implemented"; }

  /**
   * @brief Construct Function. Install all the workers.
   * @param[in] threadNum Number of the threads, must greater than 1.
   */
  explicit AsyncThreadPool(size_t threadNum) {
    CHECK_GT(threadNum, 1U);
    stopping_ = false;
    workers_.resize(threadNum);
    for (auto& worker : workers_) {
      worker.reset(new std::thread([this]() { this->run(); }));
    }
  }

  ~AsyncThreadPool() {
    if (!stopping_) {
      stop();
    }
  }

  /**
   * @brief Stop all the workers normally.
   */
  void stop() {
    stopping_ = true;
    for (size_t i = 0; i < workers_.size(); i++) {
      jobs_.enqueue([] {});
    }
    for (auto& worker : workers_) {
      worker->join();
    }
  }

  /**
   * @brief A process-wide singleton. Used as a global thread pool
   *    It should be initialized by calling
   *    AsyncThreadPool::ProcessChannel(N) first to create N threads,
   *    then call AsyncThreadPool::ProcessChannel() will get the thread pool.
   */
  static AsyncThreadPool& ProcessChannel(size_t initThreadNum = 0) {
    static std::shared_ptr<AsyncThreadPool> channel(
        new AsyncThreadPool(initThreadNum));
    return *channel;
  }

  /**
   * @brief Add a job to queue and return a std::future.
   * @note The job will be executed
   * asynchronously.
   * Call std::future::get() when the execturation result is needed;
   */
  template <class F, class... Args>
  auto addJob(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    CHECK(!stopping_) << "AsyncThreadPool is closed";
    typedef typename std::result_of<F(Args...)>::type T;

    auto task = std::make_shared<std::packaged_task<T()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto res = task->get_future();
    jobs_.enqueue([task] { (*task)(); });
    return res;
  }

  /**
   * @brief Add a batch of jobs to the queue. The main thread will be blocked
   * until these jobs are finished.
   * The results will be stored in  `results` according to `jobs` order.
   *
   * @tparam F should have a return value.
   *
   * @param[in] jobs a vector of executable objection.
   * @param[in] results a vector to store the results.
   *
   * @note *results* may need to be carefully cleared before *addBatchJobs()*.
   */
  template <class F>
  void addBatchJobs(const std::vector<F>& jobs,
                    std::vector<typename std::result_of<F()>::type>& results) {
    typedef typename std::result_of<F()>::type T;
    static_assert(!std::is_same<T, void>::value,
                  "should pass a non-void function as job");

    std::vector<std::future<T>> resFuts;
    for (const auto& job : jobs) {
      resFuts.emplace_back(addJob(job));
    }
    for (auto& fut : resFuts) {
      results.emplace_back(fut.get());
    }
  }

  /**
   * @brief Add a batch of jobs reguardless of its result.
   * @tparam F don't need to have a return value.
   * @param[in] jobs a vector of executable objection.
   */
  template <class F>
  void addBatchJobs(const std::vector<F>& jobs) {
    CHECK(!stopping_) << "AsyncThreadPool is closed";
    std::vector<std::future<bool>> tmpRes;

    for (const auto& job : jobs) {
      tmpRes.emplace_back(addJob([&job] {
        job();
        return true;
      }));
    }

    for (auto& res : tmpRes) {
      res.get();
    }
  }

protected:
  /**
   * @brief Execute the jobs in the job queue.
   */
  void run() {
    while (true) {
      JobFunc func = jobs_.dequeue();
      func();
      if (stopping_) break;
    }
  }

private:
  std::vector<std::unique_ptr<std::thread>> workers_;
  Queue<JobFunc> jobs_;
  bool stopping_;
};  // class AsyncThreadPool

}  // namespace paddle
