package pserver

/*
// TODO(zhihong): move compile flags to cmake go_library
#cgo pkg-config: protobuf
#cgo CFLAGS: -I ../../
#cgo LDFLAGS: /Users/dzh/.go/src/github.com/PaddlePaddle/Paddle/build/go/pserver/cclient/libpaddle_go_optimizer.a
#include "paddle/optimizer/optimizer.h"
*/
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
	p := paramWithConfigs.Param
	c := paramWithConfigs.Config
	buffer := &p.Content[0]
	o.opt = C.paddle_create_optimizer(C.uchar(c), C.int(len(c)), unsafe.Pointer(buffer), C.int(len(p.Content)), nullPtr, 0)
	return o
}

func (o *optimizer) GetWeights(p *Parameter) error {

	var buffer unsafe.Pointer
	buffer_len := C.paddle_optimizer_get_weights(unsafe.Pointer(o), &buffer)
	if buffer_len == 0 || buffer == nullPtr {
		return fmt.Errorf("parameter optimizer error : %s get failed", p.name)
	}
	p.Content = cArrayToSlice(buffer, int(buffer_len))
	return nil
}

func (o *optimizer) UpdateParameter(g Gradient) error {
	if o.ElementType != g.ElementType {
		return fmt.Errorf("Name: %s, parameter and gradient element type not match, parameter: %v, gradient: %v", g.Name, g.ElementType, g.ElementType)
	}

	// FIXME: do we need a copy? discard g.Content by GC ok
	r := C.paddle_update_parameter(o.opt, C.paddle_element_type(g.ElementType), unsafe.Pointer(g.Content), C.int(len(g.Content)))
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
