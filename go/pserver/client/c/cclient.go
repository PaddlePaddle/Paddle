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

package main

/*
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
  unsigned char*      content;
  int                 content_len;
} paddle_parameter, paddle_gradient;

typedef int paddle_pserver_client;
#define PSERVER_ERROR -1
#define PSERVER_OK 0
*/
import "C"

import (
	"strings"
	"sync"
	"unsafe"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	"github.com/PaddlePaddle/Paddle/go/pserver/client"
	log "github.com/inconshreveable/log15"
)

func init() {
	log.Root().SetHandler(
		log.LvlFilterHandler(log.LvlWarn, log.CallerStackHandler("%+v", log.StderrHandler)),
	)
}

var mu sync.Mutex
var handleMap = make(map[C.paddle_pserver_client]*client.Client)
var curHandle C.paddle_pserver_client

func add(c *client.Client) C.paddle_pserver_client {
	mu.Lock()
	defer mu.Unlock()
	cli := curHandle
	curHandle++
	handleMap[cli] = c
	return cli
}

func get(client C.paddle_pserver_client) *client.Client {
	mu.Lock()
	defer mu.Unlock()
	return handleMap[client]
}

func remove(client C.paddle_pserver_client) *client.Client {
	mu.Lock()
	defer mu.Unlock()
	h := handleMap[client]
	delete(handleMap, client)
	return h
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

type selector bool

func (s selector) Select() (bool, error) {
	return bool(s), nil
}

func (s selector) Done() error {
	return nil
}

type lister []client.Server

func (l lister) List() []client.Server {
	return l
}

//export paddle_new_pserver_client
func paddle_new_pserver_client(addrs *C.char, selected int) C.paddle_pserver_client {
	a := C.GoString(addrs)
	as := strings.Split(a, ",")
	servers := make([]client.Server, len(as))
	for i := range as {
		servers[i].Index = i
		servers[i].Addr = as[i]
	}
	c := client.NewClient(lister(servers), len(as), selector(selected != 0))
	return add(c)
}

//export paddle_new_etcd_pserver_client
func paddle_new_etcd_pserver_client(etcdEndpoints *C.char) C.paddle_pserver_client {
	addr := C.GoString(etcdEndpoints)
	etcdClient := client.NewEtcd(addr)
	c := client.NewClient(etcdClient, etcdClient.Desired(), etcdClient)
	return add(c)
}

//export paddle_pserver_client_release
func paddle_pserver_client_release(client C.paddle_pserver_client) {
	remove(client)
}

// paddle_begin_init_params tells trainer if it needs to init the
// parameters.
//
// returns 1 if the trainer needs to init the parameters. 0 if the
// trainer does not need to init the parameters.
//
//export paddle_begin_init_params
func paddle_begin_init_params(client C.paddle_pserver_client) C.int {
	c := get(client)
	selected, err := c.BeginInitParams()
	if err != nil {
		panic(err)
	}

	if selected {
		return 1
	}
	return 0
}

//export paddle_init_param
func paddle_init_param(client C.paddle_pserver_client, param C.paddle_parameter, paramConfig unsafe.Pointer, configLen C.int) C.int {
	et := pserver.ElementType(param.element_type)
	name := C.GoString(param.name)
	content := cArrayToSlice(unsafe.Pointer(param.content), int(param.content_len))
	pc := pserver.ParameterWithConfig{
		Param:  pserver.Parameter{Name: name, ElementType: et, Content: content},
		Config: cArrayToSlice(paramConfig, int(configLen)),
	}
	c := get(client)
	err := c.InitParam(pc)

	if err != nil {
		if err.Error() == pserver.AlreadyInitialized {
			log.Warn(
				"parameter already initialized, treat paddle_init_param as successful.",
				log.Ctx{"parameter": name},
			)
			return C.PSERVER_OK
		}
		log.Error("error init param", log.Ctx{"error": err})
		return C.PSERVER_ERROR
	}

	return C.PSERVER_OK
}

//export paddle_finish_init_params
func paddle_finish_init_params(client C.paddle_pserver_client) C.int {
	c := get(client)
	err := c.FinishInitParams()
	if err != nil {
		if err.Error() == pserver.AlreadyInitialized {
			log.Warn("parameters already initialized, treat paddle_finish_init_params as successful.")
			return C.PSERVER_OK
		}

		log.Error("error finish init params", log.Ctx{"error": err})
		return C.PSERVER_ERROR
	}

	return C.PSERVER_OK
}

//export paddle_send_grads
func paddle_send_grads(client C.paddle_pserver_client, grads **C.paddle_gradient, total C.int) C.int {
	var gs []pserver.Gradient
	for i := 0; i < int(total); i++ {
		grad := *(**C.paddle_gradient)(unsafe.Pointer((uintptr(unsafe.Pointer(grads)) + uintptr(i)*unsafe.Sizeof(*grads))))
		et := pserver.ElementType(grad.element_type)
		name := C.GoString(grad.name)
		content := cArrayToSlice(unsafe.Pointer(grad.content), int(grad.content_len))
		gs = append(gs, pserver.Gradient{Name: name, ElementType: et, Content: content})
	}

	c := get(client)
	err := c.SendGrads(gs)
	if err != nil {
		log.Error("error send grads", log.Ctx{"error": err})
		return C.PSERVER_ERROR
	}

	return C.PSERVER_OK
}

//export paddle_get_params
func paddle_get_params(client C.paddle_pserver_client, dst **C.paddle_parameter, total C.int) C.int {
	var ns []string
	for i := 0; i < int(total); i++ {
		param := *(**C.paddle_parameter)(unsafe.Pointer((uintptr(unsafe.Pointer(dst)) + uintptr(i)*unsafe.Sizeof(*dst))))
		ns = append(ns, C.GoString(param.name))
	}
	c := get(client)
	ps, err := c.GetParams(ns)
	if err != nil {
		log.Error("error get params", log.Ctx{"error": err})
		return C.PSERVER_ERROR
	}

	if len(ps) != len(ns) {
		pn := make([]string, len(ps))
		for i, p := range ps {
			pn[i] = p.Name
		}
		log.Error(
			"pserver returned wrong number of parameters.",
			log.Ctx{
				"Requested": strings.Join(pn, ", "),
				"Returned":  strings.Join(ns, ", "),
			},
		)
		return C.PSERVER_ERROR
	}

	for i := range ps {
		if ns[i] != ps[i].Name {
			pn := make([]string, len(ps))
			for i, p := range ps {
				pn[i] = p.Name
			}
			log.Error(
				"pserver returned wrong parameters, or not in requested order.",
				log.Ctx{
					"Requested": strings.Join(pn, ", "),
					"Returned":  strings.Join(ns, ", "),
				},
			)
			return C.PSERVER_ERROR
		}
	}

	for i := 0; i < int(total); i++ {
		p := ps[i]
		param := *(**C.paddle_parameter)(unsafe.Pointer((uintptr(unsafe.Pointer(dst)) + uintptr(i)*unsafe.Sizeof(*dst))))

		if unsafe.Pointer(param) == nil {
			log.Error("must pre-allocate parameter.")
			return C.PSERVER_ERROR
		}

		if unsafe.Pointer(param.content) != nil {
			if int(param.content_len) != len(p.Content) {
				log.Error(
					"the pre-allocated content len does not match parameter content len.",
					log.Ctx{
						"Pre-allocated len": param.content_len,
						"Returned len":      len(p.Content),
					},
				)
				return C.PSERVER_ERROR
			}
		}

		C.memcpy(unsafe.Pointer(param.content), unsafe.Pointer(&p.Content[0]), C.size_t(len(p.Content)))
		param.content_len = C.int(len(p.Content))
		param.element_type = C.paddle_element_type(p.ElementType)
	}

	return C.PSERVER_OK
}

func main() {} // Required but ignored
