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
package main

import "../paddle"
import "strings"
import "io/ioutil"
import "strconv"
import "reflect"

func main() {
	config := paddle.NewAnalysisConfig()
	config.SetModel("data/model/__model__", "data/model/__params__")
    config.DisableGlogInfo()
    config.SwitchUseFeedFetchOps(false)
    config.SwitchSpecifyInputNames(true)

    predictor := paddle.NewPredictor(config)

    println("============== paddle inference ==============")
    println("input num: ", predictor.GetInputNum())
    println("input name: ", predictor.GetInputNames()[0])
    println("output num: ", predictor.GetOutputNum())
    println("output name: ", predictor.GetInputNames()[0])
    println("============== run inference =================")

    input := predictor.GetInputTensors()[0]
    output := predictor.GetOutputTensors()[0]

    filename := "data/data.txt"
    data := ReadData(filename)
    input.SetValue(data[:1 * 3 * 300 * 300])
    input.Reshape([]int32{1, 3, 300, 300})

    predictor.SetZeroCopyInput(input)
    predictor.ZeroCopyRun()
    predictor.GetZeroCopyOutput(output)

    println("============= parse output ===================")
    output_val := output.Value()
    value := reflect.ValueOf(output_val)
    shape, dtype := paddle.ShapeAndTypeOf(value)
    switch dtype {
    case paddle.PaddleDType(paddle.FLOAT32):
        v := value.Interface().([][]float32)
        println("v: ", v[0][0], v[0][1], "...")
    case paddle.PaddleDType(paddle.UINT8):
        v := value.Interface().([][]uint8)
        println("v: ", v[0][0], v[0][1], "...")
    case paddle.PaddleDType(paddle.INT32):
        v := value.Interface().([][]int32)
        println("v: ", v[0][0], v[0][1], "...")
    case paddle.PaddleDType(paddle.INT64):
        v := value.Interface().([][]int64)
        println("v: ", v[0][0], v[0][1], "...")
    }
    println(shape[0], shape[1])
    println(output.Shape()[0])
}

func ReadData(filename string) []float32 {
    file_bytes, _ := ioutil.ReadFile(filename)
    data_slice := strings.Split(string(file_bytes), " ")
    var result []float32
    for _, n := range data_slice {
        r, _ := strconv.ParseFloat(n, 32)
        result = append(result, float32(r))
    }
    return result
}
