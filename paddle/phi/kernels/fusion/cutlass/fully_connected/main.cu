#include "paddle/phi/kernels/fusion/cutlass/fully_connected/fc_util.h"
#include "paddle/phi/kernels/fusion/cutlass/fully_connected/fc_decl.h"
#include <dlfcn.h>
#include <random>
#include <ctime>
// namespace phi{
// namespace fusion{
// namespace cutlass_internal{

// using DataType_ = __nv_bfloat16;
using DataType_ = half;

cutlass::Status fc_bias_relu_sm80_fp16_35(const FcAllParams& params);

typedef void (*func)(FcAllParams);

void InitMatrix(DataType_ *matrix, int rows, int cols);

inline void* GetDsoHandle() {
    void* dso_handle = nullptr;
    std::string dso_path = "/tyk/Paddle/paddle/phi/kernels/fusion/cutlass/fully_connected/build/libCutlassFc.so";
    dso_handle = dlopen(dso_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    return dso_handle;
}

void EnumParamList_MNK_Act(int M_, int N_, int K_, OpType op_type, std::string activation, float leaky_alpha_=1.0){
    int M = M_;
    int N = N_;
    int K = K_;
    // 行主序
    int lda = K, ldb = N, ldd = N;
    // int ldc_bias = 0;
   
    float leaky_alpha = leaky_alpha_;
    int sm_version = 80;

    FcDataType date_type = FcDataType::fp16;
    if constexpr(std::is_same<DataType_, __nv_bfloat16>::value){
        date_type = FcDataType::bf16;
    }
    else if constexpr(std::is_same<DataType_, float>::value){
        date_type = FcDataType::fp32;
    }
    else{
        ;
    }

    ///TODO:初始化和传输
    DataType_ *input, *weight, *bias, *output;
    CUDA_CHECK(cudaMalloc((void**)&input, sizeof(DataType_) * M * K));
    CUDA_CHECK(cudaMalloc((void**)&weight, sizeof(DataType_) * K * N));
    // 行主序的bias应该是N
    CUDA_CHECK(cudaMalloc((void**)&bias, sizeof(DataType_) * N));
    CUDA_CHECK(cudaMalloc((void**)&output, sizeof(DataType_) * M * N));
    InitMatrix(input, M, K);
    InitMatrix(weight, K, N);
    InitMatrix(bias, 1, N);
    InitMatrix(output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
   
    // cutlass
    FcAllParams params{
        input, weight, bias, output, M, N, K, lda, ldb, ldd, stream, date_type, sm_version, leaky_alpha
    };
    // void* dlhandler = GetDsoHandle();
    // func fc_func = NULL;
    // if(dlhandler == NULL){
    //     throw std::runtime_error("failed to load .so dynamically");
    // }
    
    // if (activation == "fc_bias_relu") {
    //     fc_func = (func)(dlsym(dlhandler, "FcBiasRelu"));
    // } else if (activation == "fc_bias_silu") {
    //     fc_func = (func)(dlsym(dlhandler, "FcBiasSilu"));
    // } else if (activation == "fc_bias") {
    //     fc_func = (func)(dlsym(dlhandler, "FcBias"));
    // } else if (activation == "fc_bias_leaky_relu") {
    //     fc_func = (func)(dlsym(dlhandler, "FcBiasLeakyRelu"));
    // } else if (activation == "fc_bias_sigmoid") {
    //     fc_func = (func)(dlsym(dlhandler, "FcBiasSigmoid"));
    // } else {
    //     throw "Cutlass does not support current activation!";
    // }
    // fc_func(params);

    cutlass::Status status = fc_bias_relu_sm80_fp16_35(params);
    
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaDeviceSynchronize());
    

    // navie and diff
    float max_diff = fc_diff_gpu<DataType_>(params, op_type);
    std::cout << max_diff << std::endl;

    // dlclose(dlhandler);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(weight));
    CUDA_CHECK(cudaFree(bias));
    CUDA_CHECK(cudaFree(output));
}

int main(){
    std::default_random_engine e;
    std::uniform_int_distribution<unsigned> u(1, 65536);

    float leaky_alpha = 1.0;
    OpType ops[5] = {OpType::FC_BIAS,
                    OpType::FC_BIAS_RELU,
                    OpType::FC_BIAS_SILU,
                    OpType::FC_BIAS_LEAKY_RELU,
                    OpType::FC_BIAS_SIGMOID};
    ///
    /*
    // mod
    int op_idx = 0;
    std::string activation = OpType2String(ops[op_idx]);
    if(activation == "fc_bias_leaky_relu")
        leaky_alpha = 0.01;
    for(int i = 0; i < 100; i+=7){
        int M = u(e);               // 
        int N = 8*(i+1);            // u(e);
        int K = 8*(i*3+7);
        std::cout << "MNK= [" << M << ", " <<  N << ", " << K << "], Act: " << activation << std::endl;
        EnumParamList_MNK_Act(M, N, K, ops[op_idx], activation, leaky_alpha);
    }
    */
    

    /// 
    /**/
    int M = 64;
    int N = 64;
    int K = 64;
    OpType op_type = OpType::FC_BIAS_RELU;
    std::string activation = OpType2String(op_type);
    EnumParamList_MNK_Act(M, N, K, op_type, activation, 1.f);
    
    return 0;
}

__global__ void InitMatrix_kernel(DataType_ *matrix, int rows, int cols){
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    if(i < rows && j < cols){
        int offset = i*cols + j;
        const int k = 16807;
        const int m = 16;
        float valToAssign = float((offset * k % m) - m/2);
        matrix[offset] = (DataType_)(valToAssign);

        // if constexpr(std::is_same<DataType_, half>::value){
        //     matrix[offset] = __float2half(valToAssign);
        // }
        // else if constexpr(std::is_same<DataType_, __nv_bfloat16>::value){
        //     matrix[offset] = __float2bfloat16(valToAssign);
        // }
        // else if constexpr(std::is_same<DataType_, float>::value){
        //     matrix[offset] = valToAssign;
        // }
        // else{
        //     matrix[offset] = 0.;
        // }
    }
}

void InitMatrix(DataType_ *matrix, int rows, int cols){
    dim3 block(16, 16);
    dim3 grid((cols+block.x-1)/block.x, (rows+block.y-1)/block.y);
    InitMatrix_kernel<<<grid, block>>>(matrix, rows, cols);
}

// }
// }
// }