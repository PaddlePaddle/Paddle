// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef NDEBUG
#undef NDEBUG
#endif
#include "paddle/fluid/platform/engine.h"
#include <gtest/gtest.h>
#include "paddle/fluid/platform/engine_impl.h"
#include "paddle/fluid/platform/thread_utils.h"

using namespace engine;

/*
class DebugEngineTester : public ::testing::Test {
 protected:
  virtual void SetUp() {
    EngineProperty prop;
    engine = CreateEngine("DebugEngine", prop);
  }

  std::shared_ptr<Engine> engine;
};

TEST(DebugEngine, init) {
  EngineProperty prop;
  auto engine = CreateEngine("DebugEngine", prop);
}

TEST_F(DebugEngineTester, init) {}

TEST_F(DebugEngineTester, NewOperation) {
  auto fn = [](RunContext ctx, engine::CallbackOnComplete cb) {
    LOG(INFO) << "debug async fn run.";
    cb();
  };

  auto opr = engine->NewOperation(fn, {}, {});
  ASSERT_TRUE(opr);
}

TEST_F(DebugEngineTester, PushAsync) {
  auto fn = [](RunContext ctx, engine::CallbackOnComplete cb) {
    LOG(INFO) << "debug async fn run.";
    cb();
  };
  RunContext ctx;
  engine->PushAsync(fn, ctx, {}, {});
}

TEST_F(DebugEngineTester, Terminate) { engine->Terminate(); }

TEST(task_queue, multithread) {
  int count = 0;

  swiftcpp::thread::TaskQueue<int> task_queue;
  swiftcpp::thread::ThreadPool workers(1, [&] {
    while (true) {
      int task;
      bool flag = task_queue.Pop(&task);
      if (!flag) return;
      count++;
      DLOG(INFO) << task;
    }
  });

  for (int i = 0; i < 100; i++) {
    task_queue.Push(i);
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));
  task_queue.SignalForKill();
}

 */
class ThreadedResourceTester : public ::testing::Test {
 protected:
  virtual void SetUp() {
    EngineProperty prop;
    prop.num_cpu_threads = 2;
    engine = CreateEngine("MultiThreadEnginePooled", prop);
    var = engine->NewResource("resource");
  }

  ResourceHandle var;
  std::shared_ptr<Engine> engine;
};

TEST_F(ThreadedResourceTester, push_write) {
  Engine::AsyncFn fn = [](RunContext ctx, CallbackOnComplete cb) {
    LOG(INFO) << "fn run";
    cb();
    LOG(INFO) << "fn finished run";
  };

  ThreadedResourceTestHelper res_helper;

  auto res0 = engine->NewResource("res0");
  auto res1 = engine->NewResource("res1");
  auto res2 = engine->NewResource("res2");
  auto res3 = engine->NewResource("res3");
  auto res4 = engine->NewResource("res4");

  auto opr0 = engine->NewOperation(fn, {res0, res1}, {res3}, "opr0");
  auto opr1 = engine->NewOperation(fn, {res3}, {res1}, "opr1");
  auto opr2 = engine->NewOperation(fn, {res3, res1}, {res4}, "opr2");
  auto opr3 = engine->NewOperation(fn, {res3}, {}, "opr3");
  auto opr4 = engine->NewOperation(fn, {res0}, {res1, res2}, "opr4");
  auto opr5 = engine->NewOperation(fn, {res0}, {res1, res2}, "opr5");

  RunContext ctx;
  engine->PushAsync(opr0, ctx);
  ASSERT_EQ(res_helper.queue_size(res0), 0);
  engine->PushAsync(opr1, ctx);
  engine->PushAsync(opr2, ctx);
  engine->PushAsync(opr3, ctx);
  engine->PushAsync(opr4, ctx);
  engine->PushAsync(opr5, ctx);

  DLOG(INFO) << "to display resources";
  for (auto res : {res0, res1, res2, res3, res4}) {
    LOG(INFO) << res->Cast<ThreadedResource>()->debug_string();
  }

  LOG(INFO) << "engine info:\n" << engine->StatusInfo();

  engine->WaitForAllFinished();
}

class MultiThreadEnginePooledTester : public ::testing::Test {
 protected:
  virtual void SetUp() {
    EngineProperty prop;
    prop.num_cpu_threads = 2;
    engine = CreateEngine("MultiThreadEnginePooled", prop);

    var_a = engine->NewResource("a");
    var_b = engine->NewResource("b");
    var_c = engine->NewResource("c");
    var_d = engine->NewResource("d");
    var_e = engine->NewResource("e");
  }

  ResourceHandle var_a, var_b, var_c, var_d, var_e;

  std::shared_ptr<Engine> engine;
  std::map<std::string, ResourceHandle> vars;
};

TEST_F(MultiThreadEnginePooledTester, NewOperation) {
  Engine::AsyncFn fn = [](RunContext ctx, CallbackOnComplete cb) {
    LOG(INFO) << "async fn run";
    cb();
  };
  auto read_vars = std::vector<ResourceHandle>{var_a, var_b};
  auto write_vars = std::vector<ResourceHandle>{var_c};
  auto opr = engine->NewOperation(fn, read_vars, write_vars);
}

TEST_F(MultiThreadEnginePooledTester, PushAsync) {
  bool flag = false;
  Engine::AsyncFn fn = [&](RunContext ctx, CallbackOnComplete cb) {
    LOG(INFO) << "async fn run";
    flag = true;
    LOG(INFO) << "flog: " << flag;
    cb();
  };
  auto read_vars = std::vector<ResourceHandle>{var_a, var_b};
  auto write_vars = std::vector<ResourceHandle>{var_c};
  auto opr = engine->NewOperation(fn, read_vars, write_vars);
  RunContext ctx;
  LOG(INFO) << "PushAsync";
  engine->PushAsync(opr, ctx);
  LOG(INFO) << "WaitForAllFinished";
  engine->WaitForAllFinished();
  ASSERT_TRUE(flag);
}

TEST_F(MultiThreadEnginePooledTester, PushAsync_Order) {
  Engine::AsyncFn fn = [&](RunContext ctx, CallbackOnComplete cb) {
    LOG(INFO) << "async fn run";
    cb();
  };
  // profile::Profiler::Get()->Clear(); // functions are
  //   func0: A = B + 1
  //   func1: C = A + 1
  //   func2: D = B + A
  //   func3: B = D
  // order:
  //   func0
  //   func1, func2
  //   func3
  auto func0 = engine->NewOperation(fn, {var_b}, {var_c}, "func0");
  auto func1 = engine->NewOperation(fn, {var_a}, {var_c}, "func1");
  auto func2 = engine->NewOperation(fn, {var_b, var_a}, {var_d}, "func2");
  auto func3 = engine->NewOperation(fn, {var_d}, {var_b}, "func3");

  engine::RunContext ctx;
  engine->PushAsync(func0, ctx);
  engine->PushAsync(func1, ctx);
  engine->PushAsync(func2, ctx);
  engine->PushAsync(func3, ctx);

  /*
#if USE_PROFILE
  DLOG(INFO) << profile::Profiler::Get()->debug_string();
#endif
   */

  for (auto var :
       std::vector<ResourceHandle>{var_a, var_b, var_c, var_d, var_e}) {
    DLOG(INFO) << var->Cast<ThreadedResource>()->debug_string();
  }

  engine->WaitForAllFinished();
}

// Randomly generate
TEST_F(MultiThreadEnginePooledTester, CreazyTest) {
  std::vector<ResourceHandle> vars;
  const int var_num = 5;
  for (int i = 0; i < var_num; i++) {
    vars.push_back(engine->NewResource("var_" + std::to_string(i)));
  }

  for (int i = 0; i < 1000; i++) {
    Engine::AsyncFn fn = [](RunContext ctx, CallbackOnComplete cb) { cb(); };
    std::vector<ResourceHandle> read_vars;
    std::vector<ResourceHandle> write_vars;
    std::stringstream ss;
    std::set<int> ids;
    ss << "reads: ";
    for (int i = 0; i < std::rand() % 10 + 1; i++) {
      int id = rand() % var_num;
      if (ids.count(id) == 0) {
        read_vars.push_back(vars[id]);
        ids.insert(id);
        ss << vars[id]->Cast<ThreadedResource>()->name() << " ";
      }
    }
    ss << "\twrites: ";
    for (int i = 0; i < std::rand() % 10 + 1; i++) {
      int id = rand() % var_num;
      if (ids.count(id) == 0) {
        write_vars.push_back(vars[id]);
        ids.insert(id);
        ss << vars[id]->Cast<ThreadedResource>()->name() << " ";
      }
    }

    engine::RunContext ctx;

    std::string func_name = "func_" + std::to_string(i);
    auto opr = engine->NewOperation(fn, read_vars, write_vars, func_name);
    DLOG(INFO) << func_name << "\t" << ss.str();
    engine->PushAsync(opr, ctx);
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));
  for (auto var : vars) {
    DLOG(INFO) << var->Cast<ThreadedResource>()->debug_string();
  }
  engine->WaitForAllFinished();
}
