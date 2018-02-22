// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pserver

// #cgo CFLAGS: -I ../../
// #cgo LDFLAGS: ${SRCDIR}/client/c/libpaddle_go_optimizer.a -lstdc++ -lm
// #include "paddle/optimizer/optimizer.h"
// #include <stdlib.h>
// #include <string.h>
import "C"

import (
	"fmt"
	"unsafe"

	log "github.com/inconshreveable/log15"
)

type optimizer struct {
	opt         *C.struct_paddle_optimizer
	elementType ElementType
	contentLen  int
	config      []byte
}

func cArrayToSlice(p unsafe.Pointer, len int) []byte {
	if p == nil {
		return nil
	}

	// create a Go clice backed by a C array, reference:
	// https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	//
	// Go garbage collector will not interact with this data, need
	// to be freed properly.
	return (*[1 << 30]byte)(p)[:len:len]
}

func newOptimizer(paramWithConfigs ParameterWithConfig, State []byte) *optimizer {
	o := &optimizer{}
	o.elementType = paramWithConfigs.Param.ElementType
	o.contentLen = len(paramWithConfigs.Param.Content)
	p := paramWithConfigs.Param
	c := paramWithConfigs.Config
	s := State
	paramBufferSize := C.size_t(len(p.Content))
	log.Info("New Optimizer Created with config", log.Ctx{
		"ElementType": p.ElementType,
		"ParamSize":   paramBufferSize,
		"ConfigSize":  len(c),
		"StateSize":   len(s),
	})
	var cbuffer unsafe.Pointer
	cbuffer = C.malloc(paramBufferSize)

	C.memcpy(cbuffer, unsafe.Pointer(&p.Content[0]), paramBufferSize)
	var cstate unsafe.Pointer
	if len(s) != 0 {
		cstate = unsafe.Pointer(&s[0])
	}

	var cptr (*C.uchar)
	if len(c) > 0 {
		cptr = (*C.uchar)(&c[0])
	} else {
		log.Error("empty config", "param name", paramWithConfigs.Param.Name)
	}
	o.config = c
	o.opt = C.paddle_create_optimizer(
		cptr,
		C.int(len(c)),
		C.paddle_element_type(p.ElementType),
		cbuffer,
		C.int(paramBufferSize),
		(*C.char)(cstate),
		C.int(len(s)),
	)
	return o
}

func (o *optimizer) GetWeights() []byte {
	var buffer unsafe.Pointer
	// we do not own the buffer, no need to free later.
	bufferLen := C.paddle_optimizer_get_weights(o.opt, &buffer)
	return cArrayToSlice(buffer, int(bufferLen)*C.sizeof_float)
}

func (o *optimizer) GetStates() []byte {
	var cbuffer *C.char
	// we owns the state buffer, need to free later.
	cbufferLen := C.paddle_optimizer_get_state(o.opt, &cbuffer)
	buf := cArrayToSlice(unsafe.Pointer(cbuffer), int(cbufferLen))
	cpy := make([]byte, len(buf))
	copy(cpy, buf)
	C.free(unsafe.Pointer(cbuffer))
	return cpy
}

func (o *optimizer) UpdateParameter(g Gradient) error {
	if o.elementType != g.ElementType {
		return fmt.Errorf("Name: %s, parameter and gradient element type not match, parameter: %v, gradient: %v", g.Name, o.elementType, g.ElementType)
	}

	if o.contentLen != len(g.Content) {
		return fmt.Errorf("Name: %s, parameter and gradient does not have same content len, parameter: %d, gradient: %d", g.Name, o.contentLen, len(g.Content))
	}

	r := C.paddle_update_parameter(o.opt, C.paddle_element_type(g.ElementType), unsafe.Pointer(&g.Content[0]), C.int(len(g.Content)))
	if r != 0 {
		return fmt.Errorf("optimizer update returned error code: %d", r)
	}
	return nil
}

func (o *optimizer) Cleanup() {
	if unsafe.Pointer(o.opt) != nil {
		C.paddle_release_optimizer(o.opt)
		o.opt = (*C.struct_paddle_optimizer)(nil)
	}
}
