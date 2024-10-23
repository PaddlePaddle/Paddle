/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <numeric>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/inference/api/paddle_infer_contrib.h"
#include "paddle/phi/common/float16.h"
#include "test/cpp/inference/api/trt_test_helper.h"

namespace paddle_infer {

class InferApiTesterUtils {
 public:
  static std::unique_ptr<Tensor> CreateInferTensorForTest(
      const std::string &name, PlaceType place, void *p_scope) {
    auto var = static_cast<paddle::framework::Scope *>(p_scope)->Var(name);
    var->GetMutable<phi::DenseTensor>();
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    const auto &dev_ctxs = pool.device_contexts();
    std::unique_ptr<Tensor> res(new Tensor(p_scope, &dev_ctxs));
    res->input_or_output_ = true;
    res->SetName(name);
    res->SetPlace(place, 0 /*device id*/);
    return res;
  }
};

TEST(Tensor, copy_to_cpu_async_stream) {
  LOG(INFO) << GetVersion();
  UpdateDllFlag("conv_workspace_size_limit", "4000");
  std::string model_dir = FLAGS_infer_model + "/model";
  Config config;
  config.EnableNewIR(false);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableUseGpu(100, 0);

  auto predictor = CreatePredictor(config);
  auto pred_clone = predictor->Clone();

  std::vector<int> in_shape = {1, 3, 318, 318};
  int in_num =
      std::accumulate(in_shape.begin(), in_shape.end(), 1, [](int &a, int &b) {
        return a * b;
      });

  std::vector<float> input(in_num, 1.0);

  const auto &input_names = predictor->GetInputNames();
  auto input_tensor = predictor->GetInputHandle(input_names[0]);

  input_tensor->Reshape(in_shape);
  input_tensor->CopyFromCpu(input.data());

  predictor->Run();

  const auto &output_names = predictor->GetOutputNames();
  auto output_tensor = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_tensor->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  float *out_data = static_cast<float *>(
      contrib::TensorUtils::CudaMallocPinnedMemory(sizeof(float) * out_num));
  memset(out_data, 0, sizeof(float) * out_num);
  std::vector<float> correct_out_data = {
      127.78,
      1.07353,
      -229.42,
      1127.28,
      -177.365,
      -292.412,
      -271.614,
      466.054,
      540.436,
      -214.223,
  };

  for (int i = 0; i < 100; i++) {
    predictor->Run();
  }

  cudaStream_t stream;
  output_tensor->CopyToCpuAsync(out_data, static_cast<void *>(&stream));

  // sync
  cudaStreamSynchronize(stream);

  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(out_data[i] / correct_out_data[i], 1.0, 1e-3);
  }
  contrib::TensorUtils::CudaFreePinnedMemory(static_cast<void *>(out_data));
}

TEST(Tensor, copy_to_cpu_async_callback) {
  LOG(INFO) << GetVersion();
  UpdateDllFlag("conv_workspace_size_limit", "4000");
  std::string model_dir = FLAGS_infer_model + "/model";
  Config config;
  config.SwitchIrOptim(false);
  config.EnableNewIR(false);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableUseGpu(100, 0);

  auto predictor = CreatePredictor(config);
  auto pred_clone = predictor->Clone();

  std::vector<int> in_shape = {1, 3, 318, 318};
  int in_num =
      std::accumulate(in_shape.begin(), in_shape.end(), 1, [](int &a, int &b) {
        return a * b;
      });

  std::vector<float> input(in_num, 1.0);

  const auto &input_names = predictor->GetInputNames();
  auto input_tensor = predictor->GetInputHandle(input_names[0]);

  input_tensor->Reshape(in_shape);
  input_tensor->CopyFromCpu(input.data());

  predictor->Run();

  const auto &output_names = predictor->GetOutputNames();
  auto output_tensor = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_tensor->shape();
  int out_num = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  float *out_data = static_cast<float *>(
      contrib::TensorUtils::CudaMallocPinnedMemory(sizeof(float) * out_num));
  memset(out_data, 0, sizeof(float) * out_num);

  for (int i = 0; i < 100; i++) {
    predictor->Run();
  }
  cudaDeviceSynchronize();

  output_tensor->CopyToCpuAsync(
      out_data,
      [](void *cb_params) {
        float *data = static_cast<float *>(cb_params);
        std::vector<float> correct_out_data = {
            127.78,
            1.07353,
            -229.42,
            1127.28,
            -177.365,
            -292.412,
            -271.614,
            466.054,
            540.436,
            -214.223,
        };
        for (int i = 0; i < 10; i++) {
          EXPECT_NEAR(data[i] / correct_out_data[i], 1.0, 1e-3);
        }
      },
      static_cast<void *>(out_data));

  cudaDeviceSynchronize();
  contrib::TensorUtils::CudaFreePinnedMemory(static_cast<void *>(out_data));
}

template <class DTYPE>
static void test_copy_tensor(PlaceType src_place, PlaceType dst_place) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", src_place, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", dst_place, static_cast<void *>(&scope));
  std::vector<DTYPE> data_src(6, 1);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<DTYPE> data_dst(4, 2);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(tensor_dst.get(), *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<DTYPE> data_check(6, 3);
  tensor_dst->CopyToCpu<DTYPE>(static_cast<DTYPE *>(data_check.data()));

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(data_check[i], 1, 1e-5);
  }
}

TEST(CopyTensor, float64) {
  test_copy_tensor<double>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<double>(PlaceType::kCPU, PlaceType::kGPU);
  test_copy_tensor<double>(PlaceType::kGPU, PlaceType::kCPU);
  test_copy_tensor<double>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, float32) {
  test_copy_tensor<float>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<float>(PlaceType::kCPU, PlaceType::kGPU);
  test_copy_tensor<float>(PlaceType::kGPU, PlaceType::kCPU);
  test_copy_tensor<float>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, int32) {
  test_copy_tensor<int32_t>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<int32_t>(PlaceType::kCPU, PlaceType::kGPU);
  test_copy_tensor<int32_t>(PlaceType::kGPU, PlaceType::kCPU);
  test_copy_tensor<int32_t>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, int64) {
  test_copy_tensor<int64_t>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<int64_t>(PlaceType::kCPU, PlaceType::kGPU);
  test_copy_tensor<int64_t>(PlaceType::kGPU, PlaceType::kCPU);
  test_copy_tensor<int64_t>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, int8) {
  test_copy_tensor<int8_t>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<int8_t>(PlaceType::kCPU, PlaceType::kGPU);
  test_copy_tensor<int8_t>(PlaceType::kGPU, PlaceType::kCPU);
  test_copy_tensor<int8_t>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, uint8) {
  test_copy_tensor<uint8_t>(PlaceType::kCPU, PlaceType::kCPU);
  test_copy_tensor<uint8_t>(PlaceType::kCPU, PlaceType::kGPU);
  test_copy_tensor<uint8_t>(PlaceType::kGPU, PlaceType::kCPU);
  test_copy_tensor<uint8_t>(PlaceType::kGPU, PlaceType::kGPU);
}

TEST(CopyTensor, bool_cpu_to_cpu) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));

  std::array<bool, 6> data_src;
  data_src.fill(true);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::array<bool, 4> data_dst;
  data_dst.fill(false);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(tensor_dst.get(), *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::array<bool, 6> data_check;
  data_check.fill(false);
  tensor_dst->CopyToCpu<bool>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == true);
  }
}

TEST(CopyTensor, bool_gpu_to_gpu) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kGPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kGPU, static_cast<void *>(&scope));

  std::array<bool, 6> data_src;
  data_src.fill(true);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::array<bool, 4> data_dst;
  data_dst.fill(false);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(tensor_dst.get(), *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::array<bool, 6> data_check;
  data_check.fill(false);
  tensor_dst->CopyToCpu<bool>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == true);
  }
}

TEST(CopyTensor, bool_gpu_to_cpu) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kGPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));

  std::array<bool, 6> data_src;
  data_src.fill(true);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::array<bool, 4> data_dst;
  data_dst.fill(false);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(tensor_dst.get(), *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::array<bool, 6> data_check;
  data_check.fill(false);
  tensor_dst->CopyToCpu<bool>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == true);
  }
}

TEST(CopyTensor, bool_cpu_to_gpu) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kGPU, static_cast<void *>(&scope));

  std::array<bool, 6> data_src;
  data_src.fill(true);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::array<bool, 4> data_dst;
  data_dst.fill(false);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(tensor_dst.get(), *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::array<bool, 6> data_check{false};
  data_check.fill(false);
  tensor_dst->CopyToCpu<bool>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == true);
  }
}

TEST(CopyTensor, float16_cpu_to_cpu) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));

  using phi::dtype::float16;
  std::vector<float16> data_src(6, float16(1.0));
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float16> data_dst(4, float16(2.0));
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(tensor_dst.get(), *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<float16> data_check(6, float16(2.0));
  tensor_dst->CopyToCpu<float16>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == float16(1.0));
  }
}

TEST(CopyTensor, float16_gpu_to_gpu) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kGPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kGPU, static_cast<void *>(&scope));

  using phi::dtype::float16;
  std::vector<float16> data_src(6, float16(1.0));
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float16> data_dst(4, float16(2.0));
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(tensor_dst.get(), *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<float16> data_check(6, float16(2.0));
  tensor_dst->CopyToCpu<float16>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == float16(1.0));
  }
}

TEST(CopyTensor, float16_cpu_to_gpu) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kGPU, static_cast<void *>(&scope));

  using phi::dtype::float16;
  std::vector<float16> data_src(6, float16(1.0));
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float16> data_dst(4, float16(2.0));
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(tensor_dst.get(), *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<float16> data_check(6, float16(2.0));
  tensor_dst->CopyToCpu<float16>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == float16(1.0));
  }
}

TEST(CopyTensor, float16_gpu_to_cpu) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kGPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kCPU, static_cast<void *>(&scope));

  using phi::dtype::float16;
  std::vector<float16> data_src(6, float16(1.0));
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float16> data_dst(4, float16(2.0));
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensor(tensor_dst.get(), *tensor_src);

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  std::vector<float16> data_check(6, float16(2.0));
  tensor_dst->CopyToCpu<float16>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_TRUE(data_check[i] == float16(1.0));
  }
}

TEST(CopyTensor, async_stream) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kGPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kGPU, static_cast<void *>(&scope));

  std::vector<float> data_src(6, 1.0);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float> data_dst(4, 2.0);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  cudaStream_t stream;
  paddle_infer::contrib::TensorUtils::CopyTensorAsync(
      tensor_dst.get(), *tensor_src, static_cast<void *>(&stream));

  EXPECT_EQ(tensor_dst->shape().size(), (size_t)2);
  EXPECT_EQ(tensor_dst->shape()[0], 2);
  EXPECT_EQ(tensor_dst->shape()[1], 3);

  cudaStreamSynchronize(stream);

  std::vector<float> data_check(6, 1.0);
  tensor_dst->CopyToCpu<float>(data_check.data());

  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(data_check[i], static_cast<float>(1.0), 1e-5);
  }
}

TEST(CopyTensor, async_callback) {
  paddle::framework::Scope scope;
  auto tensor_src = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_src", PlaceType::kCPU, static_cast<void *>(&scope));
  auto tensor_dst = paddle_infer::InferApiTesterUtils::CreateInferTensorForTest(
      "tensor_dst", PlaceType::kGPU, static_cast<void *>(&scope));

  std::vector<float> data_src(6, 1.0);
  tensor_src->Reshape({2, 3});
  tensor_src->CopyFromCpu(data_src.data());

  std::vector<float> data_dst(4, 2.0);
  tensor_dst->Reshape({2, 2});
  tensor_dst->CopyFromCpu(data_dst.data());

  paddle_infer::contrib::TensorUtils::CopyTensorAsync(
      tensor_dst.get(),
      *tensor_src,
      [](void *cb_params) {
        Tensor *tensor = static_cast<Tensor *>(cb_params);
        EXPECT_EQ(tensor->shape().size(), (size_t)2);
        EXPECT_EQ(tensor->shape()[0], 2);
        EXPECT_EQ(tensor->shape()[1], 3);
      },
      static_cast<void *>(&(*tensor_dst)));

  cudaDeviceSynchronize();
}

}  // namespace paddle_infer
