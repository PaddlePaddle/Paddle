// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/plugin/batched_gemm_op_plugin.h"

namespace plf = paddle::platform;
namespace dyl = paddle::platform::dynload;
namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template<typename T>
__global__ void Add_ReluKernel(T *pInput, const T* pBias, int m, int n, int bc)
{
    //inplace
    const int tx = threadIdx.x, index = blockIdx.x * m * n + threadIdx.x;
    if(index >= bc * m * n) return;
    int offset =  blockIdx.x * m *n + threadIdx.x;
    T _x = pInput[offset];
    T _y = _x + pBias[offset];
    pInput[offset] = fmaxf(0, _y);
    return;
}

template<typename T>
__global__ void AddKernel(T *pInput, const T* pBias, int m, int n, int bc)
{
    //inplace
    const int tx = threadIdx.x, index = blockIdx.x * m * n + threadIdx.x;
    if(index >= bc * m * n) return;
    int offset= blockIdx.x * m * n + threadIdx.x;
    T _x = pInput[offset];
    T _y = _x + pBias[offset];
    pInput[offset] = _y;
    return;
}

template<typename T>
void BatchedGemmKernel(const int nBatch, const int nK, const int nN, const int batchCount, const T* inputs, const T* weight, const cublasHandle_t& handle, T* outputs) {
    const T alpha = 1.0;
    const T beta = 0.0;

    int strideA = nBatch * nK;
    int strideB = nK * nN;
    int strideC = nBatch * nN;

    cudaDataType_t dataType;
    cublasComputeType_t computeType;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    if(sizeof(T) == sizeof(float)) {
    	dataType = CUDA_R_32F;
    	computeType = CUBLAS_COMPUTE_32F;
    } else {
        dataType= CUDA_R_16F;
        computeType = CUBLAS_COMPUTE_16F;
    }

    dyl::cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        nN, nBatch, nK, 
        (const void*)(&alpha), 
        (const void*)(weight), 
        dataType, nN, strideB, 
        (const void*)(inputs), 
        dataType, nK, strideA, 
        (const void*)(&beta), 
        (void*)(outputs), 
        dataType, nN, strideC, 
        batchCount, computeType, algo);
    return;
}

// class CuBLASBatchedGemmPlugin
CuBLASBatchedGemmPlugin::CuBLASBatchedGemmPlugin(const std::string &name, const nvinfer1::Weights& weight, const nvinfer1::Weights& bias, const int k, const int n, const int batchcount, const bool has_relu, const bool with_fp16):
    name_(name), nK_(k), nN_(n), batchcount_(batchcount), has_relu_(has_relu), with_fp16_(with_fp16)
{
    WHERE_AM_I()
    assert(weight.type == nvinfer1::DataType::kFLOAT || weight.type == nvinfer1::DataType::kHALF);
    assert(weight.values != nullptr);
    assert(weight.count == k * n * batchcount);
    assert(bias.type == nvinfer1::DataType::kFLOAT || weight.type == nvinfer1::DataType::kHALF);
    assert(bias.values != nullptr);
    assert(bias.count == n * batchcount);


    //if (weight.type == nvinfer1::DataType::kFLOAT)
    //    std::cout << "bg pluin init::float" << std::endl;
    //if (weight.type == nvinfer1::DataType::kHALF)
    //    std::cout << "bg pluin init::half" << std::endl;
    nvinfer1::DataType mType = with_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; 
    
    weight_.convertAndCopy(weight, mType);
    void* cuda_mem{nullptr};
    int64_t nb_bytes = weight_.count * getElementSize(mType);
    cudaMalloc(&cuda_mem, nb_bytes);
    cudaMemcpy(cuda_mem, weight_.values, nb_bytes, cudaMemcpyHostToDevice);
    pGPUWeight_.reset(cuda_mem);
    
    bias_.convertAndCopy(bias, mType);
    void* cuda_mem2{nullptr};
    int64_t nb_bytes2 = bias_.count * getElementSize(mType);
    cudaMalloc(&cuda_mem2, nb_bytes2);
    cudaMemcpy(cuda_mem2, bias_.values, nb_bytes2, cudaMemcpyHostToDevice);
    pGPUBias_.reset(cuda_mem2);

    dyl::cublasCreate(&handle_);
}

CuBLASBatchedGemmPlugin::CuBLASBatchedGemmPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name) 
{
    WHERE_AM_I()
    const char *data   = reinterpret_cast<const char *>(buffer);
    size_t      offset = 0;
    memcpy(&nK_, data + offset, sizeof(nK_));
    offset += sizeof(nK_);
    memcpy(&nN_, data + offset, sizeof(nN_));
    offset += sizeof(nN_);
    memcpy(&batchcount_, data + offset, sizeof(batchcount_));
    offset += sizeof(batchcount_);
    memcpy(&has_relu_, data + offset, sizeof(has_relu_));
    offset += sizeof(has_relu_);
    memcpy(&with_fp16_, data + offset, sizeof(with_fp16_));
    offset += sizeof(with_fp16_);

    nvinfer1::DataType mType = with_fp16_ ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; 
    weight_.type   = mType;
    weight_.count  = nK_ * nN_ * batchcount_;
    size_t size    = getElementSize(mType) * nK_ * nN_ * batchcount_;
    weight_.values = malloc(size);
    memcpy(reinterpret_cast<char *>(const_cast<void *>(weight_.values)), data + offset, size);
    offset += size;

    bias_.type   = mType;
    bias_.count  = nN_ * batchcount_;
    size_t size_bias    = getElementSize(mType) *  nN_ * batchcount_;
    bias_.values = malloc(size_bias);
    memcpy(reinterpret_cast<char *>(const_cast<void *>(bias_.values)), data + offset, size_bias);

    void* cuda_mem{nullptr};
    int64_t nb_bytes = weight_.count * getElementSize(mType);
    cudaMalloc(&cuda_mem, nb_bytes);
    cudaMemcpy(cuda_mem, weight_.values, nb_bytes, cudaMemcpyHostToDevice);
    pGPUWeight_.reset(cuda_mem);
    
    void* cuda_mem2{nullptr};
    int64_t nb_bytes2 = bias_.count * getElementSize(mType);
    cudaMalloc(&cuda_mem2, nb_bytes2);
    cudaMemcpy(cuda_mem2, bias_.values, nb_bytes2, cudaMemcpyHostToDevice);
    pGPUBias_.reset(cuda_mem2);

    dyl::cublasCreate(&handle_);
}

CuBLASBatchedGemmPlugin::~CuBLASBatchedGemmPlugin()
{
    WHERE_AM_I();
}

nvinfer1::IPluginV2DynamicExt *CuBLASBatchedGemmPlugin::clone() const noexcept {
    WHERE_AM_I()
    CuBLASBatchedGemmPlugin *p = new CuBLASBatchedGemmPlugin(name_, weight_, bias_, nK_, nN_, batchcount_, has_relu_, with_fp16_);
    p->setPluginNamespace(namespace_.c_str());
    return p;
}

int32_t CuBLASBatchedGemmPlugin::getNbOutputs() const noexcept {
    WHERE_AM_I()
    return 1;
}

nvinfer1::DataType CuBLASBatchedGemmPlugin::getOutputDataType(int32_t index, nvinfer1::DataType const *inputTypes, int32_t nbInputs) const noexcept
{
    WHERE_AM_I()
    assert(inputTypes[0] == nvinfer1::DataType::kFLOAT || inputTypes[0] == nvinfer1::DataType::kHALF);
    return inputTypes[0];
}

nvinfer1::DimsExprs CuBLASBatchedGemmPlugin::getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs *inputs, int32_t nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept
{
    WHERE_AM_I()
    nvinfer1::DimsExprs ret {inputs[0]};
    ret.d[inputs[0].nbDims - 1] = exprBuilder.constant(nN_);
    return ret;
}

bool CuBLASBatchedGemmPlugin::supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    WHERE_AM_I()
    switch (pos)
    {
    case 0:
        if (with_fp16_) {
            return (inOut[0].type == nvinfer1::DataType::kHALF) &&
                (inOut[0].format == nvinfer1::TensorFormat::kLINEAR);
        }
        else {
            return (inOut[0].type == nvinfer1::DataType::kFLOAT) &&
                (inOut[0].format == nvinfer1::TensorFormat::kLINEAR);
        }
    case 1:
        return inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
    default: // should NOT be here!
        return false;
    }
    return false;
}

void CuBLASBatchedGemmPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int32_t nbInputs, const nvinfer1::DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
{
    WHERE_AM_I();
}

size_t
CuBLASBatchedGemmPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
{
    WHERE_AM_I()
    return 0;
}

int32_t CuBLASBatchedGemmPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
    WHERE_AM_I()

    dyl::cublasSetStream(handle_, stream);
    auto t = inputDesc[0].type;
    /*
    if(t == nvinfer1::DataType::kFLOAT)
      std::cout<< "***enqueue::input::float***" << std::endl;
    if(t == nvinfer1::DataType::kHALF)
      std::cout<< "***enqueue::input::half***" << std::endl;

    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        std::cout<<"dim "<<i << " : "<<inputDesc[0].dims.d[i]<<std::endl;
    }
    */

    int nBatch = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i) {
        nBatch *= inputDesc[0].dims.d[i];
    }
    nBatch /= batchcount_;
    //std::cout << "batched m:" << nBatch << std::endl;

    int grid = batchcount_;
    int thread = nBatch * nN_;

    if(with_fp16_ && t == nvinfer1::DataType::kHALF) {
        //std::cout << "*****enqueue with fp16*******" << std::endl;
        const half* input = (const half*)(inputs[0]); 
        const half* weight = (const half*)(pGPUWeight_.get());
        const half* bias = (const half*)(pGPUBias_.get());
        half* output = (half*)(outputs[0]);
        BatchedGemmKernel<half>(nBatch, nK_, nN_, batchcount_, input, weight, handle_, output);
        if(has_relu_) {
            Add_ReluKernel<half><<<grid, thread>>>(output, bias, nBatch, nN_, batchcount_);
        } else {
            AddKernel<half><<<grid, thread>>>(output, bias, nBatch, nN_, batchcount_);
        }
    } else {
        //std::cout << "*****enqueue with fp32*******" << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const float* input = (const float*)(inputs[0]); 
        const float* weight = (const float*)(pGPUWeight_.get());
        const float* bias = (const float*)(pGPUBias_.get());
        float* output = (float*)(outputs[0]);
        BatchedGemmKernel<float>(nBatch, nK_, nN_, batchcount_, input, weight, handle_, output);
        if(has_relu_) {
            Add_ReluKernel<float><<<grid, thread>>>(output, bias, nBatch, nN_, batchcount_);
        } else {
            AddKernel<float><<<grid, thread>>>(output, bias, nBatch, nN_, batchcount_);
        }
    }
    return 0;
}

int32_t CuBLASBatchedGemmPlugin::initialize() noexcept {
    WHERE_AM_I()
    return 0;
}

void CuBLASBatchedGemmPlugin::terminate() noexcept {
    WHERE_AM_I()
}

void CuBLASBatchedGemmPlugin::destroy() noexcept {
    WHERE_AM_I();

    if(weight_.values != nullptr)
        free(const_cast<void *>(weight_.values));
    if(bias_.values != nullptr)
        free(const_cast<void *>(bias_.values));

    pGPUWeight_.reset(nullptr);
    pGPUBias_.reset(nullptr);

    dyl::cublasDestroy(handle_);
}

size_t CuBLASBatchedGemmPlugin::getSerializationSize() const noexcept {
    WHERE_AM_I()
    nvinfer1::DataType mType = with_fp16_ ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; 
    return sizeof(nK_) + sizeof(nN_) + sizeof(batchcount_) + sizeof(has_relu_) + 
	sizeof(with_fp16_) + getElementSize(mType) * weight_.count + getElementSize(mType) * bias_.count;
}

void CuBLASBatchedGemmPlugin::serialize(void *buffer) const noexcept
{
    WHERE_AM_I()
    nvinfer1::DataType mType = with_fp16_ ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; 
    char * data   = reinterpret_cast<char *>(buffer);
    size_t offset = 0;
    memcpy(data + offset, &nK_, sizeof(nK_));
    offset += sizeof(nK_);
    memcpy(data + offset, &nN_, sizeof(nN_));
    offset += sizeof(nN_);
    memcpy(data + offset, &batchcount_, sizeof(batchcount_));
    offset += sizeof(batchcount_);
    memcpy(data + offset, &has_relu_, sizeof(has_relu_));
    offset += sizeof(has_relu_);
    memcpy(data + offset, &with_fp16_, sizeof(with_fp16_));
    offset += sizeof(with_fp16_);
    size_t size = getElementSize(mType) * nK_ * nN_ * batchcount_;
    memcpy(data + offset, weight_.values, size);
    offset += size;
    size_t size_bias = getElementSize(mType) * nN_ * batchcount_;
    memcpy(data + offset, bias_.values, size_bias);
}

void CuBLASBatchedGemmPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    WHERE_AM_I()
    namespace_ = pluginNamespace;
}
const char *CuBLASBatchedGemmPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I()
    return namespace_.c_str();
}

const char *CuBLASBatchedGemmPlugin::getPluginType() const noexcept
{
    WHERE_AM_I()
    return "batchedgemmplugin";
}

const char *CuBLASBatchedGemmPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I()
    return "1";
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
