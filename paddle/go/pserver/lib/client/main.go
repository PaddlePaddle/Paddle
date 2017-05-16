package main

/*
#include <stdlib.h>
#include <string.h>
typedef enum {
  PADDLE_ELEMENT_TYPE_INT32   = 0,
  PADDLE_ELEMENT_TYPE_UINT32  = 1,
  PADDLE_ELEMENT_TYPE_INT64   = 2,
  PADDLE_ELEMENT_TYPE_UINT64  = 3,
  PADDLE_ELEMENT_TYPE_FLOAT32 = 4,
  PADDLE_ELEMENT_TYPE_FLOAT64 = 5,
} paddle_element_type;

typedef struct {
  char*               name;
  paddle_element_type element_type;
  char*               content;
  int                 content_len;
} paddle_parameter, paddle_gradient;

static inline void paddle_release_param(paddle_parameter* param) {
  if (param != NULL) {
    if (param->name != NULL) {
      free(param->name);
    }

    if (param->content != NULL) {
      free(param->content);
    }

    free(param);
  }
}

typedef int client;
*/
import "C"

import (
	"log"
	"sync"
	"unsafe"

	"github.com/PaddlePaddle/Paddle/paddle/go/pserver"
)

var nullPtr = unsafe.Pointer(uintptr(0))
var mu sync.Mutex
var handleMap = make(map[C.client]*pserver.Client)
var curHandle C.client

func add(c *pserver.Client) C.client {
	mu.Lock()
	defer mu.Unlock()
	client := curHandle
	curHandle++
	handleMap[client] = c
	return client
}

func get(client C.client) *pserver.Client {
	mu.Lock()
	defer mu.Unlock()
	return handleMap[client]
}

func remove(client C.client) *pserver.Client {
	mu.Lock()
	defer mu.Unlock()
	h := handleMap[client]
	delete(handleMap, client)
	return h
}

func cArrayToSlice(p unsafe.Pointer, len int) []byte {
	if p == nullPtr {
		return nil
	}

	// create a Go clice backed by a C array,
	// reference: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	return (*[1 << 30]byte)(p)[:len:len]
}

//export paddle_new_pserver_client
func paddle_new_pserver_client(addr *C.char) C.client {
	c := pserver.NewClient(C.GoString(addr))
	return add(c)
}

//export paddle_pserver_client_release
func paddle_pserver_client_release(client C.client) {
	c := remove(client)
	c.Cleanup()
}

//export paddle_begin_init_params
func paddle_begin_init_params(client C.client, pserver_config unsafe.Pointer, config_len C.int) C.int {
	c := get(client)
	b := cArrayToSlice(pserver_config, int(config_len))
	selected, err := c.BeginInitParams(b)
	if err != nil {
		log.Println(err)
		return -1
	}

	if selected {
		return 1
	}
	return 0
}

//export paddle_init_param
func paddle_init_param(client C.client, param C.paddle_parameter, param_config unsafe.Pointer, config_len C.int) C.int {
	et := pserver.ElementType(param.element_type)
	name := C.GoString(param.name)
	content := cArrayToSlice(unsafe.Pointer(param.content), int(param.content_len))
	pc := pserver.ParameterWithConfig{
		Param:  pserver.Parameter{Name: name, ElementType: et, Content: content},
		Config: cArrayToSlice(param_config, int(config_len)),
	}
	c := get(client)
	err := c.InitParam(pc)
	if err != nil {
		log.Println(err)
		return -1
	}

	return 0
}

//export paddle_finish_init_params
func paddle_finish_init_params(client C.client) C.int {
	c := get(client)
	err := c.FinishInitParams()
	if err != nil {
		log.Println(err)
		return -1
	}

	return 0
}

//export paddle_send_grads
func paddle_send_grads(client C.client, grads *C.paddle_gradient, total C.int) C.int {
	var gs []pserver.Gradient
	for i := 0; i < int(total); i++ {
		grad := (*C.paddle_gradient)(unsafe.Pointer((uintptr(unsafe.Pointer(grads)) + uintptr(i)*unsafe.Sizeof(*grads))))
		et := pserver.ElementType(grad.element_type)
		name := C.GoString(grad.name)
		content := cArrayToSlice(unsafe.Pointer(grad.content), int(grad.content_len))
		gs = append(gs, pserver.Gradient{Name: name, ElementType: et, Content: content})
	}

	c := get(client)
	err := c.SendGrads(gs)
	if err != nil {
		log.Println(err)
		return -1
	}

	return 0
}

//export paddle_get_params
func paddle_get_params(client C.client, names **C.char, dst **C.paddle_parameter, total C.int) C.int {
	var ns []string
	for i := 0; i < int(total); i++ {
		name := *(**C.char)(unsafe.Pointer((uintptr(unsafe.Pointer(names)) + uintptr(i)*unsafe.Sizeof(*names))))
		ns = append(ns, C.GoString(name))
	}
	c := get(client)
	ps, err := c.GetParams(ns)
	if err != nil {
		log.Println(err)
		return -1
	}

	for i := 0; i < int(total); i++ {
		if i >= len(ps) {
			break
		}

		p := ps[i]
		param := *(**C.paddle_parameter)(unsafe.Pointer((uintptr(unsafe.Pointer(dst)) + uintptr(i)*unsafe.Sizeof(*dst))))
		nameReady := false
		contentAllocated := false

		if unsafe.Pointer(param) == nullPtr {
			param = (*C.paddle_parameter)(C.calloc(1, C.size_t(unsafe.Sizeof(*param))))
		} else {
			if unsafe.Pointer(param.name) != nullPtr {
				if n := C.GoString(param.name); n != p.Name {
					log.Println("Warning: the pre-allocated parameter name does not match the parameter name, it will be freed.", n, p.Name)
					C.free(unsafe.Pointer(param.name))
				} else {
					nameReady = true
				}
			}

			if unsafe.Pointer(param.content) != nullPtr {
				if int(param.content_len) == len(p.Content) {
					contentAllocated = true
				} else {
					log.Println("Warning: the pre-allocated content len does not match parameter content len, the pre-allocated content will be freed.", param.content_len, len(p.Content))
					C.free(unsafe.Pointer(param.content))
				}
			}
		}

		if !nameReady {
			param.name = C.CString(p.Name)
		}
		if !contentAllocated {
			param.content = (*C.char)(C.malloc(C.size_t(len(p.Content))))
		}
		C.memcpy(unsafe.Pointer(param.content), unsafe.Pointer(&p.Content[0]), C.size_t(len(p.Content)))
		param.content_len = C.int(len(p.Content))
		param.element_type = C.paddle_element_type(p.ElementType)
	}

	return 0
}

//export paddle_save_model
func paddle_save_model(client C.client, path *C.char) C.int {
	p := C.GoString(path)
	c := get(client)
	err := c.SaveModel(p)
	if err != nil {
		log.Println(err)
		return -1
	}

	return 0
}

func main() {} // Required but ignored
