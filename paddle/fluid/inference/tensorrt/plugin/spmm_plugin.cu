/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "paddle/fluid/inference/tensorrt/plugin/spmm_plugin.h"
// #include "serialize.hpp"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

PluginFieldCollection SpmmPluginDynamicCreator::field_collection_{};
std::vector<PluginField> SpmmPluginDynamicCreator::plugin_attr_;

REGISTER_TENSORRT_PLUGIN(SpmmPluginDynamicCreator);

inline int getElementSize(DataType type) {
  switch (type) {
    case DataType::kFLOAT:
      return 4;
    case DataType::kHALF:
      return 2;
    case DataType::kINT8:
      return 1;
    default:
      assert(false && "getElementSize only supports [FLOAT|HALF|INT8]");
  }
}

inline cudaDataType_t convertTrtType(DataType type) {
  switch (type) {
    case DataType::kFLOAT:
      return CUDA_R_32F;
    case DataType::kHALF:
      return CUDA_R_16F;
    case DataType::kINT8:
      return CUDA_R_8I;
    default:
      assert(false && "convertTrtType only supports [FLOAT|HALF|INT8]");
  }
}

inline void deserialize_value_size(void const** buffer, size_t* buffer_size,
                                   void* value, size_t value_size) {
  assert(*buffer_size >= value_size);
  memcpy(value, *buffer, value_size);
  reinterpret_cast<char const*&>(*buffer) += value_size;
  *buffer_size -= value_size;
}

inline float round_scale(float x) { return std::floor(x + 0.5f); }

inline void convertAndCopy(const Weights& src, DataType type, void* dest) {
  assert((src.type == DataType::kFLOAT || src.type == DataType::kHALF) &&
         "convertAndCopy only supports src type [FLOAT|HALF]");
  assert((type == DataType::kFLOAT || type == DataType::kHALF) &&
         "convertAndCopy only supports dest type [FLOAT|HALF]");
  if (type == DataType::kFLOAT) {
    if (src.type == DataType::kFLOAT) {
      std::copy_n(static_cast<const float*>(src.values), src.count,
                  static_cast<float*>(dest));
    } else {
      for (int i = 0; i < src.count; ++i) {
        static_cast<float*>(dest)[i] =
            static_cast<float>(static_cast<const __half*>(src.values)[i]);
      }
    }
  } else {
    if (src.type == DataType::kHALF) {
      std::copy_n(static_cast<const __half*>(src.values), src.count,
                  static_cast<__half*>(dest));
    } else {
      for (int i = 0; i < src.count; ++i) {
        static_cast<__half*>(dest)[i] =
            static_cast<__half>(static_cast<const float*>(src.values)[i]);
      }
    }
  }
}

SpmmPluginDynamic::cusparseLtContext::cusparseLtContext() {
  CHECK_CUSPARSE(cusparseLtInit(&handle))
}

SpmmPluginDynamic::cusparseLtContext::~cusparseLtContext() {
  CHECK_CUSPARSE(cusparseLtDestroy(&handle))
}

void SpmmPluginDynamic::cusparseLtContext::init(
    int m, int n, int k, cudaDataType_t type, void* bias_ptr,
    SpmmPluginDynamic::Activation activation) {
  /*
  1. Init matrix descriptors (matA, matB, matC)
  2. Init matrix multiplication descriptor (matmul)
  3. Set activation and bias attribute of matmul
  4. Init algorithm selection descriptor (alg_sel)
  5. Init plan descriptor (plan)
  */
  assert(!is_initialized &&
         "Descriptor should be destroyed before calling create");
  constexpr int alignment = 16;
  cusparseComputeType compute_type;
  switch (type) {
    case CUDA_R_32F:
      compute_type = CUSPARSE_COMPUTE_TF32;
      break;
    case CUDA_R_16F:
      compute_type = CUSPARSE_COMPUTE_16F;
      break;
    case CUDA_R_8I:
      compute_type = CUSPARSE_COMPUTE_32I;
      break;
    default:
      assert(false &&
             "cusparLtContext only supports data type "
             "[CUDA_R_32F|CUDA_R_16F|CUDA_R_8I]");
  }
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
      &handle, &matA, m, k, k, alignment, type, CUSPARSE_ORDER_ROW))
  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
      &handle, &matB, n, k, k, alignment, type, CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT))
  CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
      &handle, &matC, m, n, n, alignment, type, CUSPARSE_ORDER_ROW))
  CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
      &handle, &matmul, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE, &matA, &matB, &matC, &matC, compute_type))
  if (activation == SpmmPluginDynamic::Activation::kRelu) {
    int true_value = 1;
    float relu_upper_bound = std::numeric_limits<float>::max();
    float relu_threshold = 0.0f;
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_ACTIVATION_RELU, &true_value,
        sizeof(true_value)))
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND,
        &relu_upper_bound, sizeof(relu_upper_bound)))
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD,
        &relu_threshold, sizeof(relu_threshold)))
  } else if (activation == SpmmPluginDynamic::Activation::kGelu) {
    int true_value = 1;
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_ACTIVATION_GELU, &true_value,
        sizeof(true_value)))
  } else {
    assert(activation == SpmmPluginDynamic::Activation::kNone &&
           "Received unknown activation");
  }
  if (bias_ptr != nullptr) {
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_BIAS_POINTER, &bias_ptr,
        sizeof(bias_ptr)))
  }
  CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
  int alg = 0;
  CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
      &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
  CHECK_CUSPARSE(
      cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                          workspace_size))
  is_initialized = true;
}

void SpmmPluginDynamic::cusparseLtContext::setAlgo(int alg) {
  assert(is_initialized &&
         "Descriptor should be initialized before setting algorithm");
  CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
      &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg, sizeof(alg)))
  CHECK_CUSPARSE(
      cusparseLtMatmulGetWorkspace(&handle, &alg_sel, &workspace_size))
  CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                          workspace_size))
}

void SpmmPluginDynamic::cusparseLtContext::destroy() {
  assert(is_initialized);
  CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA))
  is_initialized = false;
}

void SpmmPluginDynamic::cusparseLtContext::compressMatB(
    int n, int k, cudaDataType_t type, void* src, void** dest,
    size_t* compressed_size) {
  assert(!is_initialized);
  assert(*dest == nullptr);
  constexpr int alignment = 16;
  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
      &handle, &matB, n, k, k, alignment, type, CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT))
  CHECK_CUSPARSE(
      cusparseLtSpMMACompressedSize2(&handle, &matB, compressed_size))
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**> dest, *compressed_size))
  CHECK_CUSPARSE(cusparseLtSpMMACompress2(
      &handle, &matB, 0, CUSPARSE_OPERATION_TRANSPOSE, src, *dest, nullptr))
  CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB))
}

// Constructor for new plugin
SpmmPluginDynamic::SpmmPluginDynamic(const std::string& layer_name,
                                     const DataType precision,
                                     const int out_dim, const Weights& weight,
                                     const Weights& bias, Activation activation)
    : layer_name_(layer_name),
      precision_(precision),
      out_dim_(out_dim),
      k_(0),
      m_max_(0),
      is_configured_(false),
      optim_alg_(0),
      weight_scale_(1.0f),
      weight_compressed_(nullptr),
      weight_compressed_dev_(nullptr),
      compressed_size_(0),
      has_bias_(false),
      bias_(nullptr),
      bias_dev_(nullptr),
      activation_(activation) {
  /*
  1. Convert weight precision (on host)
  2. (Int8) Calculate scale and scale the weight (on host)
  3. Copy weight to device
  4. Compress the weight (on device)
  5. Copy the compressed weight to host
  6. Convert bias precision and copy (on host)
  */
  precision_size_ = getElementSize(precision);
  element_size_ = (precision_ == DataType::kINT8 ? 4 : precision_size_);
  assert(weight.count % out_dim == 0 &&
         "The size of weight should be divided by output dimension.");
  k_ = weight.count / out_dim;

  assert((weight.type == DataType::kFLOAT || weight.type == DataType::kHALF) &&
         "SpmmPluginDynamic only supports weight of type [FLOAT|HALF]");
  DataType weight_type;
  if (precision_ == DataType::kINT8) {
    weight_type = DataType::kFLOAT;
  } else {
    weight_type = precision_;
  }
  std::vector<char> weight_host(element_size_ * out_dim_ * k_);
  convertAndCopy(weight, weight_type, weight_host.data());
  void* weight_dev{nullptr};
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&weight_dev),
                        precision_size_ * out_dim_ * k_))
  if (precision == DataType::kINT8) {
    float max_weight{0.0f};
    for (int i = 0; i < weight.count; ++i) {
      float local_abs =
          std::abs(reinterpret_cast<const float*>(weight_host.data())[i]);
      max_weight = std::max(max_weight, local_abs);
    }
    weight_scale_ = max_weight / 127.0f;
    std::vector<int8_t> scale_buffer(weight.count);
    for (int i = 0; i < weight.count; ++i) {
      scale_buffer[i] = static_cast<int8_t>(
          round_scale(reinterpret_cast<const float*>(weight_host.data())[i] /
                      weight_scale_));
    }
    CHECK_CUDA(cudaMemcpy(weight_dev, scale_buffer.data(),
                          precision_size_ * weight.count,
                          cudaMemcpyHostToDevice))
  } else {
    CHECK_CUDA(cudaMemcpy(weight_dev, weight_host.data(),
                          precision_size_ * weight.count,
                          cudaMemcpyHostToDevice))
  }

  spmm_context_.compressMatB(out_dim_, k_, convertTrtType(precision_),
                             weight_dev, &weight_compressed_dev_,
                             &compressed_size_);
  weight_compressed_ = new char[compressed_size_];
  CHECK_CUDA(cudaMemcpy(weight_compressed_, weight_compressed_dev_,
                        compressed_size_, cudaMemcpyDeviceToHost))

  has_bias_ = (bias.count != 0);
  if (has_bias_) {
    if (bias.count != out_dim) {
      assert(false &&
             "The dimension of bias should be equal to output dimension");
    }
    assert(bias.type == DataType::kFLOAT &&
           "SpmmPluginDynamic only supports FLOAT bias");
    bias_ = new float[out_dim_];
    convertAndCopy(bias, DataType::kFLOAT, bias_);
  }

  CHECK_CUDA(cudaFree(weight_dev))
}

// Constructor for clone
SpmmPluginDynamic::SpmmPluginDynamic(const std::string& layer_name,
                                     const nvinfer1::DataType precision,
                                     const int out_dim, const int k,
                                     const void* weight_compressed,
                                     size_t compressed_size, const void* bias,
                                     bool is_configured, const int m_max,
                                     const int optim_alg, Activation activation)
    : layer_name_(layer_name),
      precision_(precision),
      out_dim_(out_dim),
      k_(k),
      m_max_(m_max),
      is_configured_(is_configured),
      optim_alg_(optim_alg),
      weight_scale_(1.0f),
      weight_compressed_(nullptr),
      weight_compressed_dev_(nullptr),
      compressed_size_(compressed_size),
      has_bias_(false),
      bias_(nullptr),
      bias_dev_(nullptr),
      activation_(activation) {
  /*
  1. Copy the compressed weight (on host)
  2. Copy the compressed weight to device
  3. Copy the bias (on host)
  4. (Configured) Copy the bias to device
  5. (Configured) Init cuSPARSELt descriptors
  */
  precision_size_ = getElementSize(precision);
  element_size_ = (precision_ == DataType::kINT8 ? 4 : precision_size_);
  // Each plugin has a copy of compressed weight
  weight_compressed_ = new char[compressed_size];
  std::copy_n(static_cast<const char*>(weight_compressed), compressed_size,
              static_cast<char*>(weight_compressed_));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&weight_compressed_dev_),
                        compressed_size))
  CHECK_CUDA(cudaMemcpy(weight_compressed_dev_, weight_compressed_,
                        compressed_size, cudaMemcpyHostToDevice))

  has_bias_ = (bias != nullptr);
  if (has_bias_) {
    // Each plugin has a copy of bias
    bias_ = new float[out_dim_];
    std::copy_n(static_cast<const char*>(bias), sizeof(float) * out_dim_,
                static_cast<char*>(bias_));
    if (is_configured_) {
      CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&bias_dev_),
                            sizeof(float) * out_dim_))
      CHECK_CUDA(cudaMemcpy(bias_dev_, bias_, sizeof(float) * out_dim_,
                            cudaMemcpyHostToDevice))
    }
  }

  if (is_configured_) {
    cudaDataType_t dataType = convertTrtType(precision_);
    spmm_context_.init(m_max_, out_dim_, k_, dataType, bias_dev_, activation_);
    spmm_context_.setAlgo(optim_alg_);
  }
}

SpmmPluginDynamic::SpmmPluginDynamic(const std::string name, const void* data,
                                     size_t length)
    : layer_name_(name),
      weight_compressed_(nullptr),
      weight_compressed_dev_(nullptr),
      bias_(nullptr),
      bias_dev_(nullptr) {
  deserialize_value(&data, &length, &precision_);
  deserialize_value(&data, &length, &precision_size_);
  deserialize_value(&data, &length, &element_size_);
  deserialize_value(&data, &length, &out_dim_);
  deserialize_value(&data, &length, &k_);
  deserialize_value(&data, &length, &m_max_);
  deserialize_value(&data, &length, &is_configured_);
  deserialize_value(&data, &length, &optim_alg_);
  deserialize_value(&data, &length, &weight_scale_);
  deserialize_value(&data, &length, &compressed_size_);
  deserialize_value(&data, &length, &has_bias_);
  deserialize_value(&data, &length, &activation_);

  assert(is_configured_ && "Deserialize data should be configured");

  weight_compressed_ = new char[compressed_size_];
  deserialize_value_size(&data, &length, weight_compressed_, compressed_size_);
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&weight_compressed_dev_),
                        compressed_size_))
  CHECK_CUDA(cudaMemcpy(weight_compressed_dev_, weight_compressed_,
                        compressed_size_, cudaMemcpyHostToDevice))

  if (has_bias_) {
    bias_ = new float[out_dim_];
    deserialize_value_size(&data, &length, bias_, sizeof(float) * out_dim_);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&bias_dev_),
                          sizeof(float) * out_dim_))
    CHECK_CUDA(cudaMemcpy(bias_dev_, bias_, sizeof(float) * out_dim_,
                          cudaMemcpyHostToDevice))
  }

  if (is_configured_) {
    cudaDataType_t dataType = convertTrtType(precision_);
    spmm_context_.init(m_max_, out_dim_, k_, dataType, bias_dev_, activation_);
    spmm_context_.setAlgo(optim_alg_);
  }
}

IPluginV2DynamicExt* SpmmPluginDynamic::clone() const noexcept {
  try {
    auto* p =
        new SpmmPluginDynamic(layer_name_, precision_, out_dim_, k_,
                              weight_compressed_, compressed_size_, bias_,
                              is_configured_, m_max_, optim_alg_, activation_);
    p->weight_scale_ = weight_scale_;
    p->setPluginNamespace(namespace_.c_str());

    return p;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  return nullptr;
}

DimsExprs SpmmPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs,
    IExprBuilder& exprBuilder) noexcept {
  int nbDims = inputs[0].nbDims;
  try {
    if (nbDims == 5) {
      assert(nbInputs == 1);
      assert(outputIndex == 0);
      int nbDims = inputs[0].nbDims;
      assert(inputs[0].nbDims == 5);
      assert(inputs[0].d[3]->getConstantValue() == 1);
      assert(inputs[0].d[4]->getConstantValue() == 1);

      DimsExprs ret;
      ret.nbDims = nbDims;
      ret.d[0] = inputs[0].d[0];
      ret.d[1] = inputs[0].d[1];
      ret.d[2] = exprBuilder.constant(out_dim_);
      ret.d[3] = exprBuilder.constant(1);
      ret.d[4] = exprBuilder.constant(1);
      return ret;
    } else if (nbDims == 4) {
      assert(nbInputs == 1);
      assert(outputIndex == 0);
      int nbDims = inputs[0].nbDims;
      assert(inputs[0].nbDims == 4);
      assert(inputs[0].d[2]->getConstantValue() == 1);
      assert(inputs[0].d[3]->getConstantValue() == 1);

      DimsExprs ret;
      ret.nbDims = nbDims;
      ret.d[0] = inputs[0].d[0];
      ret.d[1] = exprBuilder.constant(out_dim_);
      ret.d[2] = exprBuilder.constant(1);
      ret.d[3] = exprBuilder.constant(1);

      return ret;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  return DimsExprs{};
}

bool SpmmPluginDynamic::supportsFormatCombination(int pos,
                                                  const PluginTensorDesc* inOut,
                                                  int nbInputs,
                                                  int nbOutputs) noexcept {
  assert(nbInputs == 1);
  assert(nbOutputs == 1);

  const PluginTensorDesc& in = inOut[pos];
  if (pos == 0) {
    return (in.type == precision_) && (in.format == TensorFormat::kLINEAR);
  }
  const PluginTensorDesc& prev = inOut[pos - 1];

  return in.type == prev.type && in.format == prev.format;
}

void SpmmPluginDynamic::configurePlugin(const DynamicPluginTensorDesc* inputs,
                                        int nbInputs,
                                        const DynamicPluginTensorDesc* outputs,
                                        int nbOutputs) noexcept {
  /*
  The following steps are executed if not configured.
  1. (INT8) Scale the bias (on host)
  2. Copy the bias to device
  3. Search the optimal algorithm
  */
  try {
    assert(nbOutputs == 1);
    assert(nbInputs == 1);
    assert(precision_ == inputs[0].desc.type);
    const auto& inDims0 = inputs[0].desc.dims;
    assert(inDims0.nbDims == 5);
    assert(k_ == inDims0.d[2]);
    assert(inDims0.d[3] == 1);
    assert(inDims0.d[4] == 1);

    const int BS = inputs->max.d[0];

    // The optimal algorighm id is for m = m_max_
    // To Do: configurePlugin takes time when m is changed
    if (is_configured_) {
      return;
    }

    m_max_ = BS;
    if (has_bias_) {
      if (inputs->desc.type == DataType::kINT8) {
        for (int i = 0; i < out_dim_; ++i) {
          static_cast<float*>(bias_)[i] =
              static_cast<const float*>(bias_)[i] / outputs->desc.scale;
        }
      }
      CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&bias_dev_),
                            sizeof(float) * out_dim_))
      CHECK_CUDA(cudaMemcpy(bias_dev_, bias_, sizeof(float) * out_dim_,
                            cudaMemcpyHostToDevice))
    }
    cudaDataType_t dataType = convertTrtType(precision_);
    spmm_context_.init(m_max_, out_dim_, k_, dataType, bias_dev_, activation_);

    void* dA;
    void* dC;
    void* d_workspace;
    float alpha{1.0f};
    float beta{0.0f};
    if (precision_ == DataType::kINT8) {
      alpha = inputs->desc.scale * weight_scale_ / outputs->desc.scale;
    }
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dA),
                          m_max_ * k_ * sizeof(dataType)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dC),
                          m_max_ * out_dim_ * sizeof(dataType)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_workspace),
                          spmm_context_.workspace_size));
    CHECK_CUSPARSE(cusparseLtMatmulSearch(
        &spmm_context_.handle, &spmm_context_.plan, &alpha, dA,
        weight_compressed_dev_, &beta, dC, dC, d_workspace, nullptr, 0))
    CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(
        &spmm_context_.handle, &spmm_context_.alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID, &optim_alg_, sizeof(optim_alg_)))
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(d_workspace));

    is_configured_ = true;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}

size_t SpmmPluginDynamic::getWorkspaceSize(const PluginTensorDesc* inputs,
                                           int nbInputs,
                                           const PluginTensorDesc* outputs,
                                           int nbOutputs) const noexcept {
  return spmm_context_.workspace_size;
}

int SpmmPluginDynamic::enqueue(const PluginTensorDesc* inputDesc,
                               const PluginTensorDesc* outputDesc,
                               const void* const* inputs, void* const* outputs,
                               void* workSpace, cudaStream_t stream) noexcept {
  try {
    assert(is_configured_ && "The plugin is not configured before enqueue");
    assert(k_ == inputDesc->dims.d[2]);
    float alpha = 1.0f;
    float beta = 0.0f;
    if (inputDesc->type == DataType::kFLOAT) {
      const auto* const input = static_cast<const float*>(inputs[0]);
      auto* output = static_cast<float*>(outputs[0]);
      cusparseStatus_t status = cusparseLtMatmul(
          &spmm_context_.handle, &spmm_context_.plan, &alpha, input,
          weight_compressed_dev_, &beta, output, output, workSpace, &stream, 1);
      return status != CUSPARSE_STATUS_SUCCESS;
    } else if (inputDesc->type == DataType::kHALF) {
      const auto* const input = static_cast<const half*>(inputs[0]);
      auto* output = static_cast<half*>(outputs[0]);
      cusparseStatus_t status = cusparseLtMatmul(
          &spmm_context_.handle, &spmm_context_.plan, &alpha, input,
          weight_compressed_dev_, &beta, output, output, workSpace, &stream, 1);
      return status != CUSPARSE_STATUS_SUCCESS;
    } else if (inputDesc->type == DataType::kINT8) {
      alpha = inputDesc->scale * weight_scale_ / outputDesc->scale;
      const auto* const input = static_cast<const int8_t*>(inputs[0]);
      auto* output = static_cast<int8_t*>(outputs[0]);
      cusparseStatus_t status = cusparseLtMatmul(
          &spmm_context_.handle, &spmm_context_.plan, &alpha, input,
          weight_compressed_dev_, &beta, output, output, workSpace, &stream, 1);
      return status != CUSPARSE_STATUS_SUCCESS;
    } else {
      std::cerr
          << "Unsupported type error, expected [kHALF,kFLOAT], but received "
          << static_cast<int>(precision_) << std::endl;
      assert(false);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  return -1;
}

DataType SpmmPluginDynamic::getOutputDataType(int index,
                                              const DataType* inputTypes,
                                              int nbInputs) const noexcept {
  assert(index == 0);
  assert(nbInputs == 1);
  assert(inputTypes[0] == DataType::kFLOAT ||
         inputTypes[0] == DataType::kHALF || inputTypes[0] == DataType::kINT8);
  return inputTypes[0];
}

const char* SpmmPluginDynamic::getPluginType() const noexcept {
  return SPMM_NAME;
}

const char* SpmmPluginDynamic::getPluginVersion() const noexcept {
  return SPMM_VERSION;
}

int SpmmPluginDynamic::getNbOutputs() const noexcept { return 1; }

int SpmmPluginDynamic::initialize() noexcept { return 0; }

void SpmmPluginDynamic::terminate() noexcept {}

size_t SpmmPluginDynamic::getSerializationSize() const noexcept {
  return compressed_size_ + (has_bias_ ? sizeof(float) * out_dim_ : 0) +
         sizeof(precision_) + sizeof(precision_size_) + sizeof(element_size_) +
         sizeof(out_dim_) + sizeof(k_) + sizeof(m_max_) +
         sizeof(is_configured_) + sizeof(optim_alg_) + sizeof(weight_scale_) +
         sizeof(compressed_size_) + sizeof(has_bias_) + sizeof(activation_);
}

void SpmmPluginDynamic::serialize(void* buffer) const noexcept {
  serialize_value(&buffer, precision_);
  serialize_value(&buffer, precision_size_);
  serialize_value(&buffer, element_size_);
  serialize_value(&buffer, out_dim_);
  serialize_value(&buffer, k_);
  serialize_value(&buffer, m_max_);
  serialize_value(&buffer, is_configured_);
  serialize_value(&buffer, optim_alg_);
  serialize_value(&buffer, weight_scale_);
  serialize_value(&buffer, compressed_size_);
  serialize_value(&buffer, has_bias_);
  serialize_value(&buffer, activation_);

  char* d = static_cast<char*>(buffer);
  std::copy_n(static_cast<const char*>(weight_compressed_), compressed_size_,
              d);
  if (has_bias_) {
    d += compressed_size_;
    std::copy_n(static_cast<const char*>(bias_), out_dim_ * sizeof(float), d);
  }
}

void SpmmPluginDynamic::destroy() noexcept {
  delete[] reinterpret_cast<char*> weight_compressed_;
  CHECK_CUDA(cudaFree(weight_compressed_dev_))
  if (has_bias_) {
    CHECK_CUDA(cudaFree(bias_dev_))
  }
  if (is_configured_) {
    spmm_context_.destroy();
  }
  delete this;
}

void SpmmPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept {
  try {
    namespace_ = libNamespace;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}

const char* SpmmPluginDynamic::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

inline nvinfer1::DataType fieldTypeToDataType(
    const nvinfer1::PluginFieldType ftype) {
  switch (ftype) {
    case nvinfer1::PluginFieldType::kFLOAT32:
      return nvinfer1::DataType::kFLOAT;
    case nvinfer1::PluginFieldType::kFLOAT16:
      return nvinfer1::DataType::kHALF;
    case nvinfer1::PluginFieldType::kINT32:
      return nvinfer1::DataType::kINT32;
    case nvinfer1::PluginFieldType::kINT8:
      return nvinfer1::DataType::kINT8;
    default:
      throw std::invalid_argument(
          "No corresponding datatype for plugin field type");
  }
}

SpmmPluginDynamicCreator::SpmmPluginDynamicCreator() {
  plugin_attr_.emplace_back(
      PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
  plugin_attr_.emplace_back(
      PluginField("out_dim", nullptr, PluginFieldType::kINT32, 1));
  plugin_attr_.emplace_back(
      PluginField("weight", nullptr, PluginFieldType::kFLOAT32, 1));
  plugin_attr_.emplace_back(
      PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
  plugin_attr_.emplace_back(
      PluginField("activation_id", nullptr, PluginFieldType::kINT8, 1));

  field_collection_.nbFields = plugin_attr_.size();
  field_collection_.fields = plugin_attr_.data();
}

const char* SpmmPluginDynamicCreator::getPluginName() const noexcept {
  return SPMM_NAME;
}

const char* SpmmPluginDynamicCreator::getPluginVersion() const noexcept {
  return SPMM_VERSION;
}

const PluginFieldCollection*
SpmmPluginDynamicCreator::getFieldNames() noexcept {
  return &field_collection_;
}

IPluginV2* SpmmPluginDynamicCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept {
  try {
    int type_id = -1;
    int out_dim = 0;
    Weights weight{DataType::kFLOAT, nullptr, 0ll};
    Weights bias{DataType::kFLOAT, nullptr, 0ll};
    int activation_id = -1;

    for (int i = 0; i < fc->nbFields; i++) {
      std::string field_name(fc->fields[i].name);
      if (field_name.compare("type_id") == 0) {
        type_id = static_cast<const int*>(fc->fields[i].data)[0];
      } else if (field_name.compare("out_dim") == 0) {
        out_dim = static_cast<const int*>(fc->fields[i].data)[0];
      } else if (field_name.compare("weight") == 0) {
        weight.type = fieldTypeToDataType(fc->fields[i].type);
        weight.values = fc->fields[i].data;
        weight.count = fc->fields[i].length;
      } else if (field_name.compare("bias") == 0) {
        bias.type = fieldTypeToDataType(fc->fields[i].type);
        bias.values = fc->fields[i].data;
        bias.count = fc->fields[i].length;
      } else if (field_name.compare("activation_id") == 0) {
        activation_id = static_cast<const int*>(fc->fields[i].data)[0];
      } else {
        assert(false && "Unsupport plugin field");
      }
    }

    assert(type_id != -1 && "Invalid type id");
    assert(out_dim != 0 && "Invalid output dimension");
    assert(weight.count != 0 && "Invalid weight size");
    assert(activation_id != -1 && "Invalid activation (could be none)");

    DataType type = static_cast<DataType>(type_id);
    SpmmPluginDynamic::Activation activation =
        static_cast<SpmmPluginDynamic::Activation>(activation_id);
    return new SpmmPluginDynamic(name, type, out_dim, weight, bias, activation);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  return nullptr;
}

IPluginV2* SpmmPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  // This object will be deleted when the network is destroyed, which will
  // call SpmmPluginDynamic::destroy()
  try {
    return new SpmmPluginDynamic(name, serialData, serialLength);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
  return nullptr;
}

void SpmmPluginDynamicCreator::setPluginNamespace(
    const char* libNamespace) noexcept {
  try {
    namespace_ = libNamespace;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }
}

const char* SpmmPluginDynamicCreator::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
