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

#include <gtest/gtest.h>
#include <paddle/pserver/ParameterClient2.h>
#include <paddle/pserver/ParameterServer2.h>
#include <paddle/utils/Flags.h>
#include <paddle/utils/Util.h>

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_int32(num_gradient_servers);
DEFINE_string(server_addr, "127.0.0.1", "assign server address");
DEFINE_int32(server_cpu, 0, "assign server cpu");

class ParameterServer2Tester : public ParameterServer2 {
public:
  ParameterServer2Tester(std::string serverAddr,
                         int port,
                         int rdmaCpu = -1,
                         bool sepSendAndRecv = false)
      : ParameterServer2(serverAddr, port, rdmaCpu), client_(sepSendAndRecv) {}
  virtual ~ParameterServer2Tester() {}
  void setup() {
    CHECK(ParameterServer2::init());

    parameters_.clear();
    clientConfigs_.clear();

    clientConfigs_.resize(2);
    {
      ParameterConfig& config = clientConfigs_[0];
      config.set_name("para0");
      config.set_para_id(0);
      config.set_size(10000);
      config.set_device(-1);
      config.set_learning_rate(1.0);
      config.set_momentum(0.9);
    }

    {
      ParameterConfig& config = clientConfigs_[1];
      config.set_name("para1");
      config.set_para_id(1);
      config.set_size(5000);
      config.set_device(-1);
      config.set_learning_rate(0.5);
      config.set_momentum(0.4);
    }

    for (auto& config : clientConfigs_) {
      parameters_.emplace_back(new Parameter(config, /* useGpu= */ false));
    }

    size_t id = 0;
    for (auto& para : parameters_) {
      para->setID(id++);
    }

    CHECK(client_.init(parameters_));
    OptimizationConfig optConfig;
    optConfig.set_algorithm("async_sgd");
    optConfig.set_batch_size(100);
    optConfig.set_learning_rate(0.1);
    client_.setConfig(optConfig);
    client_.setParameter();
  }

  void setConfigTest();
  void setStatusTest();
  void sendParameterTest();
  void sendDataTest(SendDataType type, size_t size);
  void operationTest();
  void mergeBlockSegmentTest();
  void checkSegments(const BlockSegments& expected, const BlockSegments& segs);
  void waitPassFinishTest();
  void synchronizeTest();

protected:
  ParameterClient2 client_;
  vector<ParameterConfig> clientConfigs_;
  vector<ParameterPtr> parameters_;
};

std::unique_ptr<ParameterServer2Tester> g_server;

void ParameterServer2Tester::setConfigTest() {
  setup();

  for (auto& config : clientConfigs_) {
    auto it = configMap_.find(config.para_id());
    EXPECT_TRUE(it != configMap_.end());
    auto& serverConfig = it->second;
    EXPECT_EQ(config.name(), serverConfig.name());
    EXPECT_EQ(config.size(), serverConfig.size());
    EXPECT_EQ(config.learning_rate(), serverConfig.learning_rate());
    EXPECT_EQ(config.momentum(), serverConfig.momentum());
  }
}

void ParameterServer2Tester::setStatusTest() {
  setup();
  EXPECT_TRUE(client_.inStatus(PSERVER_STATUS_NOT_SET));
  client_.setStatus(PSERVER_STATUS_PARAMETER_READY);
  EXPECT_EQ(PSERVER_STATUS_PARAMETER_READY, status_);
  EXPECT_TRUE(client_.inStatus(PSERVER_STATUS_PARAMETER_READY));
}

real sumVector(const CpuVector& vec) {
  const real* data = vec.getData();
  size_t dim = vec.getSize();
  real sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    sum += data[i];
  }
  return sum;
}

void ParameterServer2Tester::sendParameterTest() {
  setup();

  client_.sendAndReceiveParameter(PSERVER_UPDATE_MODE_SET_PARAM,
                                  PARAMETER_VALUE,
                                  0,       // numSamples = 0
                                  0,       // cost = 0
                                  false);  // sendBackParameter = false

  vector<ParameterPtr> parameterCopies;

  for (auto& parameter : parameters_) {
    parameterCopies.emplace_back(
        new Parameter(parameter->getConfig(), /* useGpu= */ false));
    parameterCopies.back()
        ->getBuf(PARAMETER_VALUE)
        ->copyFrom(*parameter->getBuf(PARAMETER_VALUE));
  }

  client_.sendAndReceiveParameter(PSERVER_UPDATE_MODE_GET_PARAM,
                                  PARAMETER_VALUE,
                                  0,      // numSamples = 0
                                  0,      // cost = 0
                                  true);  // sendBackParameter = true

  for (size_t i = 0; i != parameters_.size(); ++i) {
    real* v1 = parameters_[i]->getBuf(PARAMETER_VALUE)->getData();
    real* v2 = parameterCopies[i]->getBuf(PARAMETER_VALUE)->getData();
    EXPECT_EQ(parameters_[i]->getSize(), parameterCopies[i]->getSize());
    size_t size = parameters_[i]->getSize();
    real sum1 = 0, sum2 = 0;
    for (size_t j = 0; j < size; ++j) {
      sum1 += v1[j];
      sum2 += v2[j];
    }
    EXPECT_EQ(sum1, sum2);
  }
}

void ParameterServer2Tester::sendDataTest(SendDataType type, size_t size) {
  ParameterClient2 client1(true);
  client1.init(parameters_);
  ParameterClient2 client2(true);
  client2.init(parameters_);
  ParameterClient2 client3(true);
  client3.init(parameters_);

  ThreadWorker worker1;
  ThreadWorker worker2;
  ThreadWorker worker3;

  double* testData1 = new double[size];
  double* testData2 = new double[size];
  double* testData3 = new double[size];
  double* getDataExpect = new double[size];
  double* getDataReal = new double[size];
  for (size_t i = 0; i < size; ++i) {
    testData1[i] = rand();  // NOLINT TODO(yuyang18): Use rand_r instead.
    testData2[i] = rand();  // NOLINT
    testData3[i] = rand();  // NOLINT
    getDataExpect[i] = testData1[i] + testData2[i] + testData3[i];
  }

  auto put1 = [&]() {
    LOG(INFO) << "putOwnData1 start";
    client1.putOwnData(0, type, testData1, size);
    LOG(INFO) << "putOwnData1 finish";
  };

  auto get1 = [&]() {
    LOG(INFO) << "sendData1 get all start";
    client1.getAllData(0, type, getDataReal, size);
    for (size_t i = 0; i < size; ++i) {
      CHECK_EQ(getDataReal[i], getDataExpect[i]);
    }
    LOG(INFO) << "sendData1 get all finish";
  };

  auto put2 = [&]() {
    LOG(INFO) << "putOwnData2 start";
    client2.putOwnData(1, type, testData2, size);
    LOG(INFO) << "putOwnData2 finish";
  };

  auto put3 = [&]() {
    LOG(INFO) << "putOwnData3 start";
    client3.putOwnData(2, type, testData3, size);
    LOG(INFO) << "putOwnData3 finish";
  };

  worker1.addJob(put1);
  worker1.addJob(get1);
  worker2.addJob(put2);
  worker3.addJob(put3);

  worker1.addJob(put1);
  worker2.addJob(put2);
  worker3.addJob(put3);
  worker1.addJob(get1);

  worker1.wait();
  worker2.wait();
  worker3.wait();
  free(testData1);
  free(testData2);
  free(testData3);
  free(getDataExpect);
  free(getDataReal);
}

void ParameterServer2Tester::operationTest() {
  PServerVector v1, v2;
  v1 = client_.createVector();
  EXPECT_EQ(NUM_PARAMETER_TYPES, v1.handle);

  v2 = client_.createVector();
  EXPECT_EQ(NUM_PARAMETER_TYPES + 1, v2.handle);

  PreparedOperations ops;
  ops.addOperation(PSERVER_OP_RESET, v1, (real)1);
  ops.addOperation(PSERVER_OP_RESET, v2, (real)2);

  real res1, res2, res3;
  ops.addOperation(PSERVER_OP_utv, v1, v2)(&res1);

  ops.addOperation(PSERVER_OP_au_bv, v1, v2, (real)-1, (real)1);
  ops.addOperation(PSERVER_OP_utv, v1, v2)(&res2);

  ops.addOperation(PSERVER_OP_au_bv, v1, v2, (real)-1, (real)1);
  ops.addOperation(PSERVER_OP_utv, v1, v2)(&res3);
  client_.doOperation(ops, false, false);

  EXPECT_EQ(30000, res1);
  EXPECT_EQ(15000, res2);
  EXPECT_EQ(0, res3);

  PServerMatrix m1, m2;
  m1 = client_.createMatrix(4);
  EXPECT_EQ(0, m1.handle);
  m2 = client_.createMatrix(8);
  EXPECT_EQ(1, m2.handle);

  // TODO(yuyang18): add tests for other operations OP_COPY, OP_au

  client_.releaseVector(v1);
  client_.releaseVector(v2);
  client_.releaseMatrix(m1);
  client_.releaseMatrix(m2);
}

void ParameterServer2Tester::checkSegments(const BlockSegments& expected,
                                           const BlockSegments& segs) {
  EXPECT_EQ(expected.size(), segs.size());
  if (expected.size() != segs.size()) {
    return;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i], segs[i]);
  }
}

void ParameterServer2Tester::mergeBlockSegmentTest() {
  {
    BlockSegments segs{{10, 20}, {30, 45}, {50, 70}};
    mergeSegments(&segs);
    checkSegments({{10, 20}, {30, 45}, {50, 70}}, segs);
  }
  {
    BlockSegments segs{{30, 45}, {50, 70}, {10, 20}};
    mergeSegments(&segs);
    checkSegments({{10, 20}, {30, 45}, {50, 70}}, segs);
  }
  {
    BlockSegments segs{{30, 45}, {50, 70}, {10, 30}};
    mergeSegments(&segs);
    checkSegments({{10, 45}, {50, 70}}, segs);
  }
  {
    BlockSegments segs{{30, 45}, {10, 70}, {10, 30}};
    mergeSegments(&segs);
    checkSegments({{10, 70}}, segs);
  }
  {
    BlockSegments segs{{30, 45}, {50, 70}, {10, 35}};
    mergeSegments(&segs);
    checkSegments({{10, 45}, {50, 70}}, segs);
  }
  {
    BlockSegments segs{{30, 45}, {50, 70}, {10, 60}};
    mergeSegments(&segs);
    checkSegments({{10, 70}}, segs);
  }
  {
    BlockSegments segs{{30, 45}, {50, 70}, {30, 47}};
    mergeSegments(&segs);
    checkSegments({{30, 47}, {50, 70}}, segs);
  }
}

void ParameterServer2Tester::waitPassFinishTest() {
  ParameterClient2 client1;
  ParameterClient2 client2;
  ParameterClient2 client3;

  ThreadWorker worker1;
  ThreadWorker worker2;
  ThreadWorker worker3;

  auto init1 = [&]() {
    LOG(INFO) << "init1 start";
    client1.init(parameters_);
    LOG(INFO) << "init1 finish";
  };

  auto init2 = [&]() {
    LOG(INFO) << "init2 start";
    client2.init(parameters_);
    LOG(INFO) << "init2 finish";
  };

  auto init3 = [&]() {
    LOG(INFO) << "init3 start";
    client3.init(parameters_);
    LOG(INFO) << "init3 finish";
  };

  auto update1 = [&]() {
    LOG(INFO) << "update1 start";
    client1.sendAndReceiveParameter(PSERVER_UPDATE_MODE_ADD_GRADIENT,
                                    PARAMETER_VALUE,
                                    0,      // numSamples = 0
                                    0,      // cost = 0
                                    true);  // sendBackParameter = false
    LOG(INFO) << "update1 finish";
  };

  auto wait1 = [&]() {
    LOG(INFO) << "wait1 start";
    client1.waitPassFinish();
    LOG(INFO) << "wait1 finish";
  };

  auto update2 = [&]() {
    LOG(INFO) << "update2 start";
    client2.sendAndReceiveParameter(PSERVER_UPDATE_MODE_ADD_GRADIENT,
                                    PARAMETER_VALUE,
                                    0,      // numSamples = 0
                                    0,      // cost = 0
                                    true);  // sendBackParameter = false
    LOG(INFO) << "update2 finish";
  };

  auto wait2 = [&]() {
    LOG(INFO) << "wait2 start";
    client2.waitPassFinish();
    LOG(INFO) << "wait2 finish";
  };

  auto op3 = [&]() {
    LOG(INFO) << "op3 start";
    PreparedOperations ops;
    ops.addOperation(PSERVER_OP_SGD);
    client3.doOperation(ops,
                        /* waitForGradient= */ true,
                        /* sendBackarameter= */ true);
    LOG(INFO) << "op3 finish";
  };

  worker1.addJob(init1);
  worker2.addJob(init2);
  worker3.addJob(init3);

  worker1.addJob(update1);
  worker2.addJob(update2);
  worker3.addJob(op3);

  worker3.addJob(op3);
  worker3.addJob(op3);
  worker2.addJob(update2);
  worker2.addJob(update2);
  worker1.addJob(wait1);

  worker2.addJob(wait2);
  worker3.addJob(op3);

  worker1.wait();
  worker2.wait();
  worker3.wait();

  LOG(INFO) << "Pass 1 finished";

  worker1.addJob(update1);
  worker2.addJob(update2);
  worker3.addJob(op3);

  worker1.wait();
  worker2.wait();
  worker3.wait();

  worker3.addJob(op3);
  worker3.addJob(op3);
  worker1.addJob(update1);
  worker1.addJob(wait1);
  worker2.addJob(wait2);

  worker1.wait();
  worker2.wait();
  worker3.wait();

  LOG(INFO) << "Pass 2 finished";
}

void ParameterServer2Tester::synchronizeTest() {
  ParameterClient2 client1;
  ParameterClient2 client2;

  ThreadWorker worker1;
  ThreadWorker worker2;

  FLAGS_log_period_server = 2;

  auto init1 = [&]() {
    LOG(INFO) << "init1 start";
    client1.init(parameters_);
    client1.setTrainerId(0);
    LOG(INFO) << "init1 finish";
  };

  auto init2 = [&]() {
    LOG(INFO) << "init2 start";
    client2.init(parameters_);
    client2.setTrainerId(1);
    LOG(INFO) << "init2 finish";
  };

  auto update1 = [&]() {
    LOG(INFO) << "update1 start";
    client1.sendAndReceiveParameter(PSERVER_UPDATE_MODE_ASYNC_SGD,
                                    PARAMETER_VALUE,
                                    0,      // numSamples = 0
                                    0,      // cost = 0
                                    true);  // sendBackParameter = false
    LOG(INFO) << "update1 finish";
  };

  auto wait1 = [&]() {
    LOG(INFO) << "wait1 start";
    client1.asyncFinishPass();
    LOG(INFO) << "wait1 finish";
  };

  auto update2 = [&]() {
    LOG(INFO) << "update2 start";
    client2.sendAndReceiveParameter(PSERVER_UPDATE_MODE_ASYNC_SGD,
                                    PARAMETER_VALUE,
                                    0,      // numSamples = 0
                                    0,      // cost = 0
                                    true);  // sendBackParameter = false
    LOG(INFO) << "update2 finish";
  };

  auto wait2 = [&]() {
    LOG(INFO) << "wait2 start";
    client2.asyncFinishPass();
    LOG(INFO) << "wait2 finish";
  };

  worker1.addJob(init1);
  worker2.addJob(init2);
  // call wait to reset some stats at pserver
  worker1.addJob(wait1);
  worker2.addJob(wait2);

  worker1.addJob(update1);
  worker2.addJob(update2);

  worker2.addJob(update2);
  worker2.addJob(update2);
  worker1.addJob(wait1);

  worker2.addJob(wait2);

  worker1.wait();
  worker2.wait();
  LOG(INFO) << "Pass 1 finished";

  worker1.addJob(update1);
  worker2.addJob(update2);

  worker1.wait();
  worker2.wait();

  worker1.addJob(update1);
  worker2.addJob(update2);
  worker1.addJob(update1);
  worker1.addJob(update1);
  worker1.addJob(update1);
  worker1.addJob(update1);
  worker1.addJob(update1);
  worker1.addJob(update1);
  worker1.addJob(wait1);
  worker2.addJob(wait2);

  worker1.wait();
  worker2.wait();
  LOG(INFO) << "Pass 2 finished";
}

TEST(ParameterServer2, sendParameter) { g_server->sendParameterTest(); }

TEST(ParameterServer2, setConfig) { g_server->setConfigTest(); }

TEST(ParameterServer2, setStatus) { g_server->setStatusTest(); }

TEST(ParameterServer2, operation) { g_server->operationTest(); }

TEST(ParameterServer2, mergeBlockSegment) { g_server->mergeBlockSegmentTest(); }

TEST(ParameterServer2, waitPassFinish) { g_server->waitPassFinishTest(); }

TEST(ParameterServer2, synchronize) { g_server->synchronizeTest(); }

TEST(ParameterServer2, sendData) {
  // Set gserver and pserver all 3, so that the test is sufficient.
  int oldFlagsPortsNUm = FLAGS_ports_num;
  int oldFlagsNumGradientServers = FLAGS_num_gradient_servers;
  int oldFlagsPort = FLAGS_port;
  FLAGS_ports_num = 3;
  FLAGS_num_gradient_servers = 3;
  FLAGS_port = FLAGS_port + 1;
  std::unique_ptr<ParameterServer2Tester> g_server1;
  std::unique_ptr<ParameterServer2Tester> g_server2;
  std::unique_ptr<ParameterServer2Tester> g_server3;
  if (FLAGS_rdma_tcp == "rdma") {
    g_server1.reset(new ParameterServer2Tester(
        FLAGS_server_addr, FLAGS_port, FLAGS_server_cpu));
    g_server1->start();
    g_server2.reset(new ParameterServer2Tester(
        FLAGS_server_addr, FLAGS_port + 1, FLAGS_server_cpu + 1));
    g_server2->start();
    g_server3.reset(new ParameterServer2Tester(
        FLAGS_server_addr, FLAGS_port + 2, FLAGS_server_cpu + 2));
    g_server3->start();
  } else {  // tcp
    g_server1.reset(new ParameterServer2Tester(FLAGS_server_addr, FLAGS_port));
    g_server1->start();
    g_server2.reset(
        new ParameterServer2Tester(FLAGS_server_addr, FLAGS_port + 1));
    g_server2->start();
    g_server3.reset(
        new ParameterServer2Tester(FLAGS_server_addr, FLAGS_port + 2));
    g_server3->start();
  }

  g_server2->init();
  g_server3->init();
  sleep(2);
  g_server1->setup();
  g_server1->sendDataTest(DATA_REDUCE_SUM, 1 << 24);
  sleep(2);
  g_server1->sendDataTest(DATA_REDUCE_SUM, 2);
  sleep(2);
  g_server1.reset();
  g_server2.reset();
  g_server3.reset();

  FLAGS_ports_num = oldFlagsPortsNUm;
  FLAGS_num_gradient_servers = oldFlagsNumGradientServers;
  FLAGS_port = oldFlagsPort;
}

int main(int argc, char** argv) {
  paddle::initMain(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  FLAGS_num_gradient_servers = 2;

  if (FLAGS_rdma_tcp == "rdma") {
    g_server.reset(new ParameterServer2Tester(
        FLAGS_server_addr, FLAGS_port, FLAGS_server_cpu));
  } else {
    g_server.reset(new ParameterServer2Tester(FLAGS_server_addr, FLAGS_port));
  }

  g_server->start();

  sleep(2);

  int ret = RUN_ALL_TESTS();

  g_server.reset();

  exit(ret);
}
