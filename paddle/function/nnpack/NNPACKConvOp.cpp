/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "nnpack.h"
#include "paddle/function/ConvOp.h"

DEFINE_bool(nnpack_allocate_outside,
            true,
            "Allocate and free workspace memory outside the NNPACK interface.");
DEFINE_int32(nnpack_num_threads,
             0,
             "The number of nnpack threads"
             "default: 0; 0 to disable threadpool.");

namespace paddle {

nnp_convolution_algorithm get_nnp_convolution_algorithm(
    const std::string& algorithm) {
  if (algorithm == "auto") {
    return nnp_convolution_algorithm_auto;
  } else if (algorithm == "ft8x8") {
    return nnp_convolution_algorithm_ft8x8;
  } else if (algorithm == "ft16x16") {
    return nnp_convolution_algorithm_ft16x16;
  } else if (algorithm == "wt8x8") {
    return nnp_convolution_algorithm_wt8x8;
  } else if (algorithm == "implicit-gemm") {
    return nnp_convolution_algorithm_implicit_gemm;
  } else if (algorithm == "direct") {
    return nnp_convolution_algorithm_direct;
  } else {
    return nnp_convolution_algorithm_auto;
  }
}

template <DeviceType Device>
class NNPACKConvFunction : public ConvFunctionBase {
 public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
    algorithm_ = get_nnp_convolution_algorithm(config.get<std::string>("algo"));
    transform_strategy_ = nnp_convolution_transform_strategy_compute;
    nnp_status status = nnp_initialize();
    CHECK_EQ(status, nnp_status_success);
    workspaceBuffer_ = nullptr;
    workspaceSize_ = 0;

    create_nnpack_threadpool();
  }

  ~NNPACKConvFunction() {
    if (workspaceBuffer_) {
      free(workspaceBuffer_);
    }
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const TensorShape& input = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& output = outputs[0].shape();
    checkShape(input, filter, output);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);
    check(inputs, outputs);
    const TensorShape& input = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& output = outputs[0].shape();

    size_t batchSize = input[0];
    size_t inputChannels = input[1];
    size_t inputHeight = input[2];
    size_t inputWidth = input[3];
    size_t filterHeight = getFilterHeight(filter);
    size_t filterWidth = getFilterWidth(filter);
    size_t outputChannels = output[1];
    size_t outputHeight = output[2];
    size_t outputWidth = output[3];

    nnp_size inputSize = {.width = inputWidth, .height = inputHeight};
    nnp_padding padding = {.top = (size_t)paddingH(),
                           .right = (size_t)paddingW(),
                           .bottom = (size_t)paddingH(),
                           .left = (size_t)paddingW()};
    nnp_size kernelSize = {.width = filterWidth, .height = filterHeight};
    nnp_size outputSubsampling = {.width = (size_t)strideW(),
                                  .height = (size_t)strideH()};

    float* inputData = inputs[0].data<float>();
    float* filterData = inputs[1].data<float>();
    float* outputData = outputs[0].data<float>();

    void* bufferPtr = nullptr;
    size_t* sizePtr = nullptr;
    size_t needSize;
    if (FLAGS_nnpack_allocate_outside) {
      if (batchSize == 1) {
        nnp_status status = nnp_convolution_inference(algorithm_,
                                                      transform_strategy_,
                                                      inputChannels,
                                                      outputChannels,
                                                      inputSize,
                                                      padding,
                                                      kernelSize,
                                                      outputSubsampling,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      &needSize,
                                                      nnp_activation_identity,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr);
        CHECK_EQ(status, nnp_status_success);
      } else {
        // only supports stride = 1
        CHECK_EQ(strideH(), 1);
        CHECK_EQ(strideW(), 1);
        nnp_status status = nnp_convolution_output(algorithm_,
                                                   batchSize,
                                                   inputChannels,
                                                   outputChannels,
                                                   inputSize,
                                                   padding,
                                                   kernelSize,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr,
                                                   &needSize,
                                                   nnp_activation_identity,
                                                   nullptr,
                                                   nullptr,
                                                   nullptr);
        CHECK_EQ(status, nnp_status_success);
      }

      VLOG(3) << "workspace size is " << needSize;
      if (needSize > workspaceSize_) {
        workspaceSize_ = needSize;
        if (workspaceBuffer_) {
          free(workspaceBuffer_);
        } else {
          posix_memalign(&workspaceBuffer_, 64, needSize);
        }
      }

      if (needSize) {
        bufferPtr = workspaceBuffer_;
        sizePtr = &needSize;
      }
    }

    size_t inputOffset = inputChannels / groups_ * inputHeight * inputWidth;
    size_t outputOffset = outputChannels / groups_ * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;

    if (batchSize == 1) {
      for (size_t g = 0; g < groups_; g++) {
        nnp_status status =
            nnp_convolution_inference(algorithm_,
                                      transform_strategy_,
                                      inputChannels / groups_,
                                      outputChannels / groups_,
                                      inputSize,
                                      padding,
                                      kernelSize,
                                      outputSubsampling,
                                      inputData + inputOffset * g,
                                      filterData + filterOffset * g,
                                      nullptr, /* bias */
                                      outputData + outputOffset * g,
                                      bufferPtr,
                                      sizePtr,
                                      nnp_activation_identity,
                                      nullptr,
                                      threadpool_, /* threadpool */
                                      nullptr);
        CHECK_EQ(status, nnp_status_success);
      }
    } else {
      // only supports stride = 1
      CHECK_EQ(strideH(), 1);
      CHECK_EQ(strideW(), 1);

      // TODO(hedaoyuan): There has some bug when batchSize > 1 and groups_ > 1.
      CHECK_EQ(groups_, static_cast<size_t>(1));
      nnp_status status = nnp_convolution_output(algorithm_,
                                                 batchSize,
                                                 inputChannels,
                                                 outputChannels,
                                                 inputSize,
                                                 padding,
                                                 kernelSize,
                                                 inputData,
                                                 filterData,
                                                 nullptr, /* bias */
                                                 outputData,
                                                 bufferPtr,
                                                 sizePtr,
                                                 nnp_activation_identity,
                                                 nullptr,
                                                 threadpool_, /* threadpool */
                                                 nullptr);
      CHECK_EQ(status, nnp_status_success);
    }
  }

  static void create_nnpack_threadpool() {
    if (FLAGS_nnpack_num_threads && threadpool_ == nullptr) {
      threadpool_ = pthreadpool_create(FLAGS_nnpack_num_threads);
      VLOG(3) << "Number of threads "
              << pthreadpool_get_threads_count(threadpool_);
    }
  }

 private:
  nnp_convolution_algorithm algorithm_;
  nnp_convolution_transform_strategy transform_strategy_;
  void* workspaceBuffer_;
  size_t workspaceSize_;
  static pthreadpool_t threadpool_;
};

template <DeviceType Device>
pthreadpool_t NNPACKConvFunction<Device>::threadpool_ = nullptr;

REGISTER_TYPED_FUNC(NNPACKConv, CPU, NNPACKConvFunction);

}  // namespace paddle
