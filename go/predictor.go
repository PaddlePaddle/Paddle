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

// #include "paddle_c_api.h"
import "C"

import "runtime"

type Predictor struct {
	c *C.PD_Predictor
}

func NewPredictor() *Predictor {
    c_predictor := C.PD_NewPredictor()
    config := &Predictor{c: c_predictor}
    runtime.SetFinalizer(config, (*config).finalizer)
    return config
}

func (predictor *Predictor) finalizer() {
    C.PD_DeletePredictor(predictor.c)
}

func DeletePredictor(predictor *Predictor) {
    C.PD_DeletePredictor(predictor.c)
}

func (predictor *Predictor) InputNum() int {
    return C.PD_GetInputNum(predictor.c)
}

func (predictor *Predictor) OutputNum() int {
    return C.PD_GetOutputNum(predictor.c)
}

func (predictor *Predictor) InputName(n: int) string {
    return C.PD_GetInputName(predictor.c, n)
}

func (predictor *Predictor) OutputName(n: int) string {
    return C.PD_GetOutputName(predictor.c, n)
}

func (predictor *Predictor) ZeroCopyRun() {
    C.PD_ZeroCopyRun(predictor.c)
}

