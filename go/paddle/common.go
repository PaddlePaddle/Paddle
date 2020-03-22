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

// #cgo CFLAGS: -Ipaddle_c/paddle/include
// #cgo LDFLAGS: -Lpaddle_c/paddle/lib -lpaddle_fluid_c
// #include <stdbool.h>
// #include <paddle_c_api.h>
import "C"
import "fmt"

func ConvertCBooleanToGo(b C.bool) bool {
	var c_false C.bool
	if b != c_false {
		return true
	}
	return false
}

func numel(shape []int32) int32 {
	n := int32(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

func bug(format string, args ...interface{}) error {
	return fmt.Errorf("Bug %v", fmt.Sprintf(format, args...))
}
