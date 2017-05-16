package pserver

/*
#include "optimizer.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type optimizerType int

const (
	sgd optimizerType = iota
)

var nullPtr = unsafe.Pointer(uintptr(0))

type optimizer struct {
	opt *C.paddle_optimizer
}

func newOptimizer(t optimizerType, learning_rate float64) *optimizer {
	o := &optimizer{}
	o.opt = C.paddle_create_SGD_optimizer(C.double(learning_rate))
	return o
}

func (o *optimizer) UpdateParameter(p Parameter, g Gradient) error {
	if len(p.Content) != len(g.Content) {
		return fmt.Errorf("parameter and gradient length not match, parameter: %d, gradient: %d", len(p.Content), len(g.Content))
	}

	if p.ElementType != g.ElementType {
		return fmt.Errorf("parameter and gradient element type not match, parameter: %v, gradient: %v", p.ElementType, g.ElementType)
	}

	r := C.paddle_update_parameter(o.opt, unsafe.Pointer(&p.Content[0]), C.paddle_element_type(p.ElementType), unsafe.Pointer(&g.Content[0]), C.int(len(g.Content)))
	if r != 0 {
		return fmt.Errorf("optimier returned error code: %d", r)
	}
	return nil
}

func (o *optimizer) Cleanup() {
	if unsafe.Pointer(o.opt) != nullPtr {
		C.paddle_release_optimizer(o.opt)
		o.opt = (*C.paddle_optimizer)(nullPtr)
	}
}
