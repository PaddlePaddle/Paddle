// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define PADDLE_MASTER_OK    0
#define PADDLE_MASTER_ERROR -1

typedef int paddle_master_client;
*/
import "C"

import (
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/PaddlePaddle/Paddle/go/master"
	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
)

var mu sync.Mutex
var handleMap = make(map[C.paddle_master_client]*master.Client)
var curHandle C.paddle_master_client

func add(c *master.Client) C.paddle_master_client {
	mu.Lock()
	defer mu.Unlock()
	client := curHandle
	curHandle++
	handleMap[client] = c
	return client
}

func get(client C.paddle_master_client) *master.Client {
	mu.Lock()
	defer mu.Unlock()
	return handleMap[client]
}

func remove(client C.paddle_master_client) *master.Client {
	mu.Lock()
	defer mu.Unlock()
	h := handleMap[client]
	delete(handleMap, client)
	return h
}

//export paddle_new_etcd_master_client
func paddle_new_etcd_master_client(etcdEndpoints *C.char, timeout int, bufSize int) C.paddle_master_client {
	p := C.GoString(etcdEndpoints)
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   strings.Split(p, ","),
		DialTimeout: time.Second * time.Duration(timeout),
	})
	if err != nil {
		panic(err)
	}
	ch := make(chan string, 1)
	a, err := master.GetKey(cli, master.DefaultAddrPath, timeout)
	if err != nil {
		panic(err)
	}
	ch <- a
	go master.WatchKey(cli, master.DefaultAddrPath, ch)
	c := master.NewClient(ch, bufSize)
	return add(c)
}

//export paddle_new_master_client
func paddle_new_master_client(addr *C.char, bufSize int) C.paddle_master_client {
	a := C.GoString(addr)
	ch := make(chan string, 1)
	ch <- a
	c := master.NewClient(ch, bufSize)
	return add(c)
}

//export paddle_release_master_client
func paddle_release_master_client(client C.paddle_master_client) {
	remove(client)
}

//export paddle_set_dataset
func paddle_set_dataset(client C.paddle_master_client, path **C.char, size C.int) C.int {
	c := get(client)
	var paths []string
	for i := 0; i < int(size); i++ {
		ptr := (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(path)) + uintptr(i)*unsafe.Sizeof(*path)))
		str := C.GoString(*ptr)
		paths = append(paths, str)
	}
	err := c.SetDataset(paths)
	if err != nil {
		log.Errorln(err)
		return C.PADDLE_MASTER_ERROR
	}

	return C.PADDLE_MASTER_OK
}

// return value:
//     0:ok
//    -1:error
//export paddle_next_record
func paddle_next_record(client C.paddle_master_client, record **C.uchar) C.int {
	c := get(client)
	r, err := c.NextRecord()
	if err != nil {
		// Error
		// TODO: return the type of error?
		*record = (*C.uchar)(nil)
		return -1
	}

	if len(r) == 0 {
		// Empty record
		*record = (*C.uchar)(nil)
		return 0
	}

	size := C.size_t(len(r))
	*record = (*C.uchar)(C.malloc(size))
	C.memcpy(unsafe.Pointer(*record), unsafe.Pointer(&r[0]), size)
	return C.int(size)
}

//export mem_free
func mem_free(p unsafe.Pointer) {
	// "free" may be a better name for this function, but doing so
	// will cause calling any function of this library from Python
	// ctypes hanging.
	C.free(p)
}

func main() {}
