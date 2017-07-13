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

	log "github.com/sirupsen/logrus"
)

var nullPtr = unsafe.Pointer(uintptr(0))

type optimizer struct {
	opt         *C.struct_paddle_optimizer
	elementType ElementType
}

func cArrayToSlice(p unsafe.Pointer, len int) []byte {
	if p == nullPtr {
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
	p := paramWithConfigs.Param
	c := paramWithConfigs.Config
	s := State
	paramBufferSize := C.size_t(len(p.Content) / C.sizeof_float)
	log.WithFields(log.Fields{
		"ElementType": p.ElementType,
		"ParamSize":   paramBufferSize,
		"ConfigSize":  len(c),
		"StateSize":   len(s),
	}).Info("New Optimizer Created with config:")
	var cbuffer unsafe.Pointer
	cbuffer = C.malloc(paramBufferSize)

	C.memcpy(cbuffer, unsafe.Pointer(&p.Content[0]), paramBufferSize)
	var cstate unsafe.Pointer
	if len(s) != 0 {
		cstate = unsafe.Pointer(&s[0])
	}

	o.opt = C.paddle_create_optimizer((*C.uchar)(&c[0]), C.int(len(c)),
		C.paddle_element_type(p.ElementType), cbuffer, C.int(paramBufferSize), (*C.char)(cstate), C.int(len(s)))
	return o
}

func (o *optimizer) GetWeights() []byte {
	var buffer unsafe.Pointer
	bufferLen := C.paddle_optimizer_get_weights(o.opt, &buffer)
	return cArrayToSlice(buffer, int(bufferLen)*C.sizeof_float)
}

func (o *optimizer) GetStates() []byte {
	var cbuffer *C.char
	cbufferLen := C.paddle_optimizer_get_state(o.opt, &cbuffer)
	return cArrayToSlice(unsafe.Pointer(cbuffer), int(cbufferLen))
}

func (o *optimizer) UpdateParameter(g Gradient) error {
	if o.elementType != g.ElementType {
		return fmt.Errorf("Name: %s, parameter and gradient element type not match, parameter: %v, gradient: %v", g.Name, o.elementType, g.ElementType)
	}

	r := C.paddle_update_parameter(o.opt, C.paddle_element_type(g.ElementType), unsafe.Pointer(&g.Content[0]), C.int(len(g.Content))/C.sizeof_float)
	if r != 0 {
		return fmt.Errorf("optimizer update returned error code: %d", r)
	}
	return nil
}

func (o *optimizer) Cleanup() {
	if unsafe.Pointer(o.opt) != nullPtr {
		C.paddle_release_optimizer(o.opt)
		o.opt = (*C.struct_paddle_optimizer)(nullPtr)
	}
}
