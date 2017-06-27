package pserver

// #cgo pkg-config: protobuf
// #cgo CFLAGS: -I ../../
// //FIXME: ldflags contain "build" path
// #cgo LDFLAGS: ../../build/go/pserver/cclient/libpaddle_go_optimizer.a -lstdc++
// #include "paddle/optimizer/optimizer.h"
// #include <stdlib.h>
// #include <string.h>
import "C"

import (
	"fmt"
	"unsafe"
)

var nullPtr = unsafe.Pointer(uintptr(0))

type optimizer struct {
	opt *C.struct_paddle_optimizer
	// used in GetParam, reconstruct Parameter from optimizer
	ElementType ElementType
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
	o.ElementType = paramWithConfigs.Param.ElementType
	p := paramWithConfigs.Param
	c := paramWithConfigs.Config
	var cbuffer unsafe.Pointer
	cbuffer_len := int(unsafe.Sizeof(p.Content[0])) * len(p.Content)
	cbuffer = C.malloc(C.size_t(cbuffer_len))
	C.memcpy(cbuffer, unsafe.Pointer(&p.Content[0]), C.size_t(cbuffer_len))
	o.opt = C.paddle_create_optimizer((*C.uchar)(&c[0]), C.int(len(c)),
		C.paddle_element_type(p.ElementType), cbuffer, C.int(len(p.Content)),
		(*C.char)(nullPtr), 0)
	return o
}

func (o *optimizer) GetWeights() []byte {
	var buffer unsafe.Pointer
	buffer_len := C.paddle_optimizer_get_weights(o.opt, &buffer)
	return cArrayToSlice(buffer, int(buffer_len))
}

func (o *optimizer) UpdateParameter(g Gradient) error {
	if o.ElementType != g.ElementType {
		return fmt.Errorf("Name: %s, parameter and gradient element type not match, parameter: %v, gradient: %v", g.Name, o.ElementType, g.ElementType)
	}

	r := C.paddle_update_parameter(o.opt, C.paddle_element_type(g.ElementType), unsafe.Pointer(&g.Content[0]), C.int(len(g.Content)))
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
