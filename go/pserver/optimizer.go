package pserver

/*
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
}

func newOptimizer(paramWithConfigs ParameterWithConfig) *optimizer {
	o := &optimizer{}
	p := paramWithConfigs.Param
	c := paramWithConfigs.Config
	o.opt = C.paddle_create_optimizer(C.uchar(c), C.int(len(c)), unsafe.Pointer(p.Content), c.int(p.Length), nullPtr, 0)
	return o
}

func (o *optimizer) UpdateParameter(p Parameter, g Gradient) error {
	if p.Length != g.Length {
		return fmt.Errorf("Name: %s, parameter and gradient length not match, parameter: %d, gradient: %d", p.Name, p.Length, g.Length)
	}

	if p.ElementType != g.ElementType {
		return fmt.Errorf("Name: %s, parameter and gradient element type not match, parameter: %v, gradient: %v", p.Name, p.ElementType, g.ElementType)
	}

	r := C.paddle_update_parameter(o.opt, C.paddle_element_type(p.ElementType), unsafe.Pointer(g.Content), C.int(g.Length))
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
