// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

// #include "paddle/fluid/inference/capi/c_api.h"
import "C"

type AnalysisConfig struct {
	c *C.PD_AnalysisConfig
}

func NewAnalysisConfig() *AnalysisConfig {
	c_config := C.PD_NewAnalysisConfig()
	config := &AnalysisConfig{c: c_config}
	return config
}

func (config *AnalysisConfig) SetModel(model, params str) {
	C.PD_SetModel(config.c, model, params)
}

func (config *AnalysisConfig) ModelDir() string {
	return C.PD_ModelDir(config.c)
}

func (config *AnalysisConfig) ProgFile() string {
	return C.PD_ProgFile(config.c)
}

func (config *AnalysisConfig) ParamsFile() {
	return C.PD_ParamsFile(config.c)
}

func (config *AnalysisConfig) EnableUseGpu(memory_pool_init_size_mb uint64, device_id int) {
	C.PD_EnableUseGpu(config.c, memory_pool_init_size_mb, device_id)
}

func (config *AnalysisConfig) DisableUseGpu() {
	C.PD_DisableUseGpu(config.c)
}

func (config *AnalysisConfig) UseGpu() bool {
	return C.PD_UseGpu(config.c)
}

func (config *AnalysisConfig) GpuDeviceId() int {
	return C.PD_GpuDeviceId(config.c)
}

func (config *AnalysisConfig) MemoryPoolInitSizeMb() int {
	return C.PD_MemoryPoolInitSizeMb(config.c)
}

func (config *AnalysisConfig) EnableCudnn() {
	C.PD_EnableCUDNN(config.c)
}

func (config *AnalysisConfig) CudnnEnabled() bool {
	return C.PD_CudnnEnabled(config.c)
}

func (config *AnalysisConfig) SwitchIrOptim(x bool) {
	C.PD_SwitchIrOptim(config.c, x)
}

func (config *AnalysisConfig) IrOptim() bool {
	return C.PD_IrOptim(config.c)
}

func (config *AnalysisConfig) SwitchUseFeedFetchOps(x bool) {
	C.PD_SwitchUseFeedFetchOps(config.c, x)
}

func (config *AnalysisConfig) UseFeedFetchOps() bool {
	return C.PD_UseFeedFetchOps(config.c)
}

func (config *AnalysisConfig) SwitchSpecifyInputNames(x bool) {
	C.PD_SwitchSpecifyInputNames(config.c, x)
}

func (config *AnalysisConfig) SpecifyInputName() bool {
	return C.PD_SpecifyInputName(config.c)
}

//func (config *AnalysisConfig) EnableTensorRtEngine(workspace_size int)

func (config *AnalysisConfig) TensorrtEngineEnabled() bool {
	return C.PD_TensorrtEngineEnabled(config.c)
}

func (config *AnalysisConfig) SwitchIrDebug(x bool) {
	C.PD_SwitchIrDebug(config.c, x)
}

func (config *AnalysisConfig) EnableNgraph() {
	C.PD_EnableNgraph(config.c)
}

func (config *AnalysisConfig) NgraphEnabled() bool {
	return C.PD_NgraphEnabled(config.c)
}

func (config *AnalysisConfig) EnableMkldnn() {
	C.PD_EnableMKLDNN(config.c)
}

func (config *AnalysisConfig) SetCpuMathLibraryNumThreads(n int) {
	C.PD_SetCpuMathLibraryNumThreads(config.c, n)
}

func (config *AnalysisConfig) CpuMathLibraryNumThreads() int {
	return C.PD_CpuMathLibraryNumThreads(config.c)
}

func (config *AnalysisConfig) EnableMkldnnQuantizer() {
	C.PD_EnableMkldnnQuantizer(config.c)
}

func (config *AnalysisConfig) MkldnnQuantizerEnabled() bool {
	return C.PD_MkldnnQuantizerEnabled()
}

// SetModelBuffer
// ModelFromMemory

func (config *AnalysisConfig) EnableMemoryOptim() {
	C.PD_EnableMemoryOptim(config.c)
}

func (config *AnalysisConfig) MemoryOptimEnabled() bool {
	return C.PD_MemoryOptimEnabled(config.c)
}

func (config *AnalysisConfig) EnableProfile() {
	C.PD_EnableProfile(config.c)
}

func (config *AnalysisConfig) ProfileEnabled() bool {
	return C.PD_ProfileEnabled(config.c)
}

// SetInValid
// IsValid
