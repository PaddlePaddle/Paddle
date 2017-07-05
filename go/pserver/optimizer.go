package pserver

// #cgo CFLAGS: -I ../../
// //FIXME: ldflags contain "build" path
// #cgo LDFLAGS: -lpaddle_go_optimizer -lstdc++ -lm
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

func newOptimizer(paramWithConfigs ParameterWithConfig) *optimizer {
	o := &optimizer{}
	o.elementType = paramWithConfigs.Param.ElementType
	p := paramWithConfigs.Param
	c := paramWithConfigs.Config
	log.WithFields(log.Fields{
		"ElementType": p.ElementType,
		"ParamSize":   len(p.Content),
		"ConfigSize":  len(c),
	}).Info("New Optimizer Created with config:")
	var cbuffer unsafe.Pointer
	cbuffer = C.malloc(C.size_t(len(p.Content)))
	C.memcpy(cbuffer, unsafe.Pointer(&p.Content[0]), C.size_t(len(p.Content)))
	o.opt = C.paddle_create_optimizer((*C.uchar)(&c[0]), C.int(len(c)),
		C.paddle_element_type(p.ElementType), cbuffer, C.int(len(p.Content)/C.sizeof_float),
		(*C.char)(nullPtr), 0)
	return o
}

func (o *optimizer) GetWeights() []byte {
	var buffer unsafe.Pointer
	buffer_len := C.paddle_optimizer_get_weights(o.opt, &buffer)
	return cArrayToSlice(buffer, int(buffer_len)*C.sizeof_float)
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
