// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

package paddle

// #include "pd_config.h"
// #include "pd_common.h"
// #include <stdlib.h>
// #include <string.h>
import "C"
import (
	"runtime"
	"unsafe"
)

type Precision C.PD_PrecisionType

const (
	Precision_Float32 Precision = C.PD_PRECISION_FLOAT32
	Precision_Int8    Precision = C.PD_PRECISION_INT8
	Precision_Half    Precision = C.PD_PRECISION_HALF
)

type Config struct {
	c *C.PD_Config
}

func NewConfig() *Config {
	cconfig := C.PD_ConfigCreate()
	config := &Config{c: cconfig}
	runtime.SetFinalizer(config, func(config *Config) {
		C.PD_ConfigDestroy(config.c)
	})
	return cconfig
}

///
/// \brief Set the combined model with two specific pathes for program and
/// parameters.
///
/// \param model model file path of the combined model.
/// \param params params file path of the combined model.
///
func (config *Config) SetModel(model, params string) {
	cmodel := C.CString(model)
	cparams := C.CString(params)
	C.PD_ConfigSetModel(config.c, cmodel, cparams)
	defer func() {
		C.free(unsafe.Pointer(cmodel))
		C.free(unsafe.Pointer(cparams))
	}()
}

///
/// \brief Set the no-combined model dir path.
///
/// \param modelDir model dir path.
///
func (config *Config) SetModelDir(modelDir string) {
	cmodel := C.CString(modelDir)
	C.PD_ConfigSetModelDir(config.c, cmodel)
	defer C.free(unsafe.Pointer(cmodel))
}

///
/// \brief Set the model file path of a combined model.
///
/// \param x model file path.
///
func (c *Config) SetProgFile(model string) {
	cmodel := C.CString(model)
	C.PD_ConfigSetProgFile(config.c, cmodel)
	defer C.free(unsafe.Pointer(cmodel))
}

///
/// \brief Set the params file path of a combined model.
///
/// \param x params file path.
///
func (config *Config) SetParamsFile(params string) {
	cparams := C.CString(params)
	C.PD_ConfigSetParamsFile(config.c, cparams)
	defer C.free(unsafe.Pointer(cparams))
}

///
/// \brief Set the path of optimization cache directory.
///
/// \param cacheDir the path of optimization cache directory.
///
func (config *Config) SetOptionCacheDir(cacheDir string) {
	ccacheDir := C.CString(cacheDir)
	C.PD_ConfigSetOptimCacheDir(config.c, ccacheDir)
	defer C.free(unsafe.Pointer(ccacheDir))
}

///
/// \brief Get the model directory path.
///
/// \return string The model directory path.
///
func (config *Config) ModelDir() string {
	return C.GoString(C.PD_ConfigGetModelDir(config.c))
}

///
/// \brief Get the program file path.
///
/// \return string The program file path.
///
func (config *Config) ProgFile() string {
	return C.GoString(C.PD_ConfigGetProgFile(config.c))
}

///
/// \brief Get the combined parameters file.
///
/// \return string The combined parameters file.
///
func (config *Config) ParamsFile() string {
	return C.GoString(C.PD_ConfigGetParamsFile(config.c))
}

///
/// \brief Turn off FC Padding.
///
func (config *Config) DisableFCPadding() {
	C.PD_ConfigDisableFCPadding(config.c)
}

///
/// \brief A boolean state telling whether fc padding is used.
///
/// \return bool Whether fc padding is used.
///
func (config *Config) UseFcPadding() bool {
	return convertPDBoolToGo(C.PD_ConfigUseFcPadding(config.c))
}

///
/// \brief Turn on GPU.
///
/// \param memorySize initial size of the GPU memory pool in MB.
/// \param deviceId the GPU card to use.
///
func (config *Config) EnableUseGpu(memorySize uint64, deviceId int32) {
	C.PD_ConfigEnableUseGpu(config.c, C.uint64_t(memorySize), C.int32_t(deviceId))
}

///
/// \brief Turn off GPU.
///
///
func (config *Config) DisableGpu() {
	C.PD_ConfigDisableGpu(config.c)
}

///
/// \brief Turn on XPU.
///
func (config *Config) EnableXpu(l3WorkspaceSize int) {
	C.PD_ConfigEnableXpu(config.c, C.int32(l3WorkspaceSize))
}

///
/// \brief A boolean state telling whether the GPU is turned on.
///
/// \return bool Whether the GPU is turned on.
///
func (config *Config) UseGpu() bool {
	return convertPDBoolToGo(C.PD_ConfigUseGpu(config.c))
}

///
/// \brief A boolean state telling whether the XPU is turned on.
///
/// \return bool Whether the XPU is turned on.
///
func (config *Config) UseXpu() bool {
	return convertPDBoolToGo(C.PD_ConfigUseXpu(config.c))
}

///
/// \brief Get the GPU device id.
///
/// \return int32 The GPU device id.
///
func (config *Config) GpuDeviceId() int32 {
	return int32(C.PD_ConfigGpuDeviceId(config.c))
}

///
/// \brief Get the XPU device id.
///
/// \return int32 The XPU device id.
///
func (config *Config) XpuDeviceId() int32 {
	return int32(C.PD_ConfigXpuDeviceId(config.c))
}

///
/// \brief Get the initial size in MB of the GPU memory pool.
///
/// \return int32 The initial size in MB of the GPU memory pool.
///
func (config *Config) MemoryPoolInitSizeMb() int32 {
	return int32(C.PD_ConfigMemoryPoolInitSizeMb(config.c))
}

///
/// \brief Get the proportion of the initial memory pool size compared to the
/// device.
///
/// \return float32 The proportion of the initial memory pool size.
///
func (config *Config) FractionOfGpuMemoryForPool() float32 {
	return float32(C.PD_ConfigFractionOfGpuMemoryForPool(config.c))
}

// ///
// /// \brief Turn on CUDNN.
// ///
// func (config *Config) EnableCudnn() {

// }

///
/// \brief A boolean state telling whether to use CUDNN.
///
/// \return bool Whether to use CUDNN.
///
// func (config *Config) CudnnEnabled() bool {

// }

///
/// \brief Control whether to perform IR graph optimization.
/// If turned off, the AnalysisConfig will act just like a NativeConfig.
///
/// \param x Whether the ir graph optimization is actived.
///
func (config *Config) SwitchIrOptim(x bool) {
	C.PD_ConfigSwitchIrOptim(config.c, convertGoBoolToPD(x))
}

///
/// \brief A boolean state telling whether the ir graph optimization is
/// actived.
///
/// \return bool Whether to use ir graph optimization.
///
// bool ir_optim() const { return enable_ir_optim_; }
func (config *Config) IrOptim() bool {
	return convertPDBoolToGo(C.PD_ConfigIrOptim(config.c))
}

///
/// \brief INTERNAL Determine whether to use the feed and fetch operators.
/// Just for internal development, not stable yet.
/// When ZeroCopyTensor is used, this should be turned off.
///
/// \param x Whether to use the feed and fetch operators.
///
// void SwitchUseFeedFetchOps(int x = true) { use_feed_fetch_ops_ = x; }

///
/// \brief A boolean state telling whether to use the feed and fetch
/// operators.
///
/// \return bool Whether to use the feed and fetch operators.
///
// bool use_feed_fetch_ops_enabled() const { return use_feed_fetch_ops_; }

///
/// \brief Control whether to specify the inputs' names.
/// The ZeroCopyTensor type has a name member, assign it with the
/// corresponding
/// variable name. This is used only when the input ZeroCopyTensors passed to
/// the
/// AnalysisPredictor.ZeroCopyRun() cannot follow the order in the training
/// phase.
///
/// \param x Whether to specify the inputs' names.
///
// void SwitchSpecifyInputNames(bool x = true) { specify_input_name_ = x; }

///
/// \brief A boolean state tell whether the input ZeroCopyTensor names
/// specified should
/// be used to reorder the inputs in AnalysisPredictor.ZeroCopyRun().
///
/// \return bool Whether to specify the inputs' names.
///
// bool specify_input_name() const { return specify_input_name_; }

///
/// \brief Turn on the TensorRT engine.
/// The TensorRT engine will accelerate some subgraphes in the original Fluid
/// computation graph. In some models such as resnet50, GoogleNet and so on,
/// it gains significant performance acceleration.
///
/// \param workspaceSize The memory size(in byte) used for TensorRT
/// workspace.
/// \param maxBatchSize The maximum batch size of this prediction task,
/// better set as small as possible for less performance loss.
/// \param minSubgraphSize The minimum TensorRT subgraph size needed, if a
/// subgraph is smaller than this, it will not be transferred to TensorRT
/// engine.
/// \param precision The precision used in TensorRT.
/// \param useStatic Serialize optimization information to disk for reusing.
/// \param useCalibMode Use TRT int8 calibration(post training
/// quantization).
///
func (config *Config) EnableTensorRtEngine(workspaceSize int32, maxBatchSize int32, minSubgraphSize int32,
	precision Precision, useStatic bool, useCalibMode bool) {
	C.PD_ConfigEnableTensorRtEngine(config.c, C.int32_t(maxBatchSize), C.int32_t(minSubgraphSize), precision, convertGoBoolToPD(useStatic), convertGoBoolToPD(useCalibMode))
}

///
/// \brief A boolean state telling whether the TensorRT engine is used.
///
/// \return bool Whether the TensorRT engine is used.
///
func (config *Config) TensorRtEngineEnabled() bool {
	return convertPDBoolToGo(C.PD_ConfigTensorRtEngineEnabled(config.c))
}

///
/// \brief Set min, max, opt shape for TensorRT Dynamic shape mode.
/// \param min_input_shape The min input shape of the subgraph input.
/// \param max_input_shape The max input shape of the subgraph input.
/// \param opt_input_shape The opt input shape of the subgraph input.
/// \param disable_trt_plugin_fp16 Setting this parameter to true means that
/// TRT plugin will not run fp16.
///
// void SetTRTDynamicShapeInfo(
// 	std::map<std::string, std::vector<int>> min_input_shape,
// 	std::map<std::string, std::vector<int>> max_input_shape,
// 	std::map<std::string, std::vector<int>> optim_input_shape,
// 	bool disable_trt_plugin_fp16 = false);
func (config *Config) SetTRTDynamicShapeInfo(minInputShape map[string][]int32, maxInputShape map[string][]int32,
	optimInputShape map[string][]int32, disableTrtPluginFp16 bool) {
	C.PD_ConfigSetTrtDynamicShapeInfo(config.c)
}

///
/// \brief Prevent ops running in Paddle-TRT
/// NOTE: just experimental, not an official stable API, easy to be broken.
///
// void Exp_DisableTensorRtOPs(const std::vector<std::string>& ops);
func (config *Config) DisableTensorRtOPs(ops []string) {

}

///
/// \brief Replace some TensorRT plugins to TensorRT OSS(
/// https://github.com/NVIDIA/TensorRT), with which some models's inference
/// may be more high-performance. Libnvinfer_plugin.so greater than
/// V7.2.1 is needed.
///
// void EnableTensorRtOSS();
func (config *Config) EnableTensorRtOSS() {

}

///
/// \brief A boolean state telling whether to use the TensorRT OSS.
///
/// \return bool Whether to use the TensorRT OSS.
///
// bool tensorrt_oss_enabled() { return trt_use_oss_; }
func (config *Config) TensorrtOssEnabled() bool {

}

///
/// \brief Enable TensorRT DLA
/// \param dla_core ID of DLACore, which should be 0, 1,
///        ..., IBuilder.getNbDLACores() - 1
///
// void EnableTensorRtDLA(int dla_core = 0);
func (config *Config) EnableTensorRtDLA(dla_core int) {

}

///
/// \brief A boolean state telling whether to use the TensorRT DLA.
///
/// \return bool Whether to use the TensorRT DLA.
///
// bool tensorrt_dla_enabled() { return trt_use_dla_; }
func (config *Config) TensorrtDlaEnabled() bool {

}

///
/// \brief Turn on the usage of Lite sub-graph engine.
///
/// \param precision_mode Precion used in Lite sub-graph engine.
/// \param passes_filter Set the passes used in Lite sub-graph engine.
/// \param ops_filter Operators not supported by Lite.
///
// void EnableLiteEngine(
// 	AnalysisConfig::Precision precision_mode = Precision::kFloat32,
// 	bool zero_copy = false,
// 	const std::vector<std::string>& passes_filter = {},
// 	const std::vector<std::string>& ops_filter = {});
func (config *Config) EnableLiteEngine(precision Precision, zero_copy bool, passes_filter []string, ops_filter []string) {

}

///
/// \brief A boolean state indicating whether the Lite sub-graph engine is
/// used.
///
/// \return bool whether the Lite sub-graph engine is used.
///
// bool lite_engine_enabled() const { return use_lite_; }
func (config *Config) LiteEngineEnabled() bool {

}

///
/// \brief Control whether to debug IR graph analysis phase.
/// This will generate DOT files for visualizing the computation graph after
/// each analysis pass applied.
///
/// \param x whether to debug IR graph analysis phase.
///
// void SwitchIrDebug(int x = true);
func (config *Config) SwitchIrDebug(x bool) {

}

///
/// \brief Turn on MKLDNN.
///
///
// void EnableMKLDNN();
func (config *Config) EnableMKLDNN() {

}

///
/// \brief Set the cache capacity of different input shapes for MKLDNN.
/// Default value 0 means not caching any shape.
/// Please see MKL-DNN Data Caching Design Document:
/// https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md
///
/// \param capacity The cache capacity.
///
// void SetMkldnnCacheCapacity(int capacity);
func (config *Config) SetMkldnnCacheCapacity(capacity int) {

}

///
/// \brief A boolean state telling whether to use the MKLDNN.
///
/// \return bool Whether to use the MKLDNN.
///
// bool mkldnn_enabled() const { return use_mkldnn_; }
func (config *Config) MkldnnEnabled() bool {

}

///
/// \brief Set the number of cpu math library threads.
///
/// \param cpu_math_library_num_threads The number of cpu math library
/// threads.
///
// void SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads);
func (config *Config) SetCpuMathLibraryNumThreads(cpu_math_library_num_threads int) {

}

///
/// \brief An int state telling how many threads are used in the CPU math
/// library.
///
/// \return int The number of threads used in the CPU math library.
///
// int cpu_math_library_num_threads() const {
// return cpu_math_library_num_threads_;
// }
func (config *Config) CpuMathLibraryNumThreads() int {

}

///
/// \brief Transform the AnalysisConfig to NativeConfig.
///
/// \return NativeConfig The NativeConfig transformed.
///
// NativeConfig ToNativeConfig() const;

///
/// \brief Specify the operator type list to use MKLDNN acceleration.
///
/// \param op_list The operator type list.
///
// void SetMKLDNNOp(std::unordered_set<std::string> op_list) {
// mkldnn_enabled_op_types_ = op_list;
// }
func (config *Config) SetMKLDNNOp(op_list []string) {

}

///
/// \brief Turn on MKLDNN quantization.
///
///
// void EnableMkldnnQuantizer();
func (config *Config) EnableMkldnnQuantizer() {

}

///
/// \brief Turn on MKLDNN bfloat16.
///
///
// void EnableMkldnnBfloat16();
func (config *Config) EnableMkldnnBfloat16() {

}

///
/// \brief A boolean state telling whether to use the MKLDNN Bfloat16.
///
/// \return bool Whether to use the MKLDNN Bfloat16.
///
// bool mkldnn_bfloat16_enabled() const { return use_mkldnn_bfloat16_; }
func (config *Config) MkldnnBfloat16Enabled() bool {

}

/// \brief Specify the operator type list to use Bfloat16 acceleration.
///
/// \param op_list The operator type list.
///
// void SetBfloat16Op(std::unordered_set<std::string> op_list) {
// bfloat16_enabled_op_types_ = op_list;
// }
func (config *Config) SetBfloat16Op(op_list []string) {

}

///
/// \brief A boolean state telling whether the thread local CUDA stream is
/// enabled.
///
/// \return bool Whether the thread local CUDA stream is enabled.
///
// bool thread_local_stream_enabled() const { return thread_local_stream_; }
func (config *Config) ThreadLocalStreamEnabled() bool {

}

///
/// \brief A boolean state telling whether the MKLDNN quantization is enabled.
///
/// \return bool Whether the MKLDNN quantization is enabled.
///
// bool mkldnn_quantizer_enabled() const { return use_mkldnn_quantizer_; }
func (config *Config) MkldnnQuantizerEnabled() bool {

}

///
/// \brief Get MKLDNN quantizer config.
///
/// \return MkldnnQuantizerConfig* MKLDNN quantizer config.
///
// MkldnnQuantizerConfig* mkldnn_quantizer_config() const;
func (config *Config) MkldnnQuantizerConfig() *MkldnnQuantizerConfig {

}

///
/// \brief Specify the memory buffer of program and parameter.
/// Used when model and params are loaded directly from memory.
///
/// \param prog_buffer The memory buffer of program.
/// \param prog_buffer_size The size of the model data.
/// \param params_buffer The memory buffer of the combined parameters file.
/// \param params_buffer_size The size of the combined parameters data.
///
// void SetModelBuffer(const char* prog_buffer, size_t prog_buffer_size,
// 					const char* params_buffer, size_t params_buffer_size);
// todo
func (config *Config) SetModelBuffer(prog_buffer, params_buffer string) {

}

///
/// \brief A boolean state telling whether the model is set from the CPU
/// memory.
///
/// \return bool Whether model and params are loaded directly from memory.
///
// bool model_from_memory() const { return model_from_memory_; }
func (config *Config) ModelFromMemory() bool {

}

///
/// \brief Turn on memory optimize
/// NOTE still in development.
///
// void EnableMemoryOptim();
func (config *Config) EnableMemoryOptim() {

}

///
/// \brief A boolean state telling whether the memory optimization is
/// activated.
///
/// \return bool Whether the memory optimization is activated.
///
// bool enable_memory_optim() const;
func (config *Config) MemoryOptimEnabled() bool {

}

///
/// \brief Turn on profiling report.
/// If not turned on, no profiling report will be generated.
///
// void EnableProfile();
func (config *Config) EnableProfile() {

}

///
/// \brief A boolean state telling whether the profiler is activated.
///
/// \return bool Whether the profiler is activated.
///
// bool profile_enabled() const { return with_profile_; }
func (config *Config) ProfileEnabled() {

}

///
/// \brief Mute all logs in Paddle inference.
///
// void DisableGlogInfo();
func (config *Config) DisableGlogInfo() {

}

///
/// \brief A boolean state telling whether logs in Paddle inference are muted.
///
/// \return bool Whether logs in Paddle inference are muted.
///
// bool glog_info_disabled() const { return !with_glog_info_; }
func (config *Config) GlogInfoDisabled() bool {

}

///
/// \brief Set the AnalysisConfig to be invalid.
/// This is to ensure that an AnalysisConfig can only be used in one
/// AnalysisPredictor.
///
// void SetInValid() const { is_valid_ = false; }
func (config *Config) SetInValid() {

}

///
/// \brief A boolean state telling whether the AnalysisConfig is valid.
///
/// \return bool Whether the AnalysisConfig is valid.
///
// bool is_valid() const { return is_valid_; }
func (config *Config) IsValid() bool {

}

///
/// \brief Get a pass builder for customize the passes in IR analysis phase.
/// NOTE: Just for developer, not an official API, easy to be broken.
///
///
// PassStrategy* pass_builder() const;
func (config *Config) PassBuilder() *PassStrategy {

}

///
/// \brief Enable the GPU multi-computing stream feature.
/// NOTE: The current behavior of this interface is to bind the computation
/// stream to the thread, and this behavior may be changed in the future.
///
// void EnableGpuMultiStream();
func (config *Config) EnableGpuMultiStream() {

}

// void PartiallyRelease();
// todo
func (config *Config) PartiallyRelease() {

}
