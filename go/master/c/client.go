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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#define PADDLE_MASTER_OK    0
#define PADDLE_MASTER_ERROR -1

#define PADDLE_SAVE_MODEL_OK   1
#define PADDLE_SAVE_MODEL_SKIP 0

typedef int paddle_master_client;
*/
import "C"

import (
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/PaddlePaddle/Paddle/go/master"
	log "github.com/inconshreveable/log15"
)

var mu sync.Mutex
var handleMap = make(map[C.paddle_master_client]*master.Client)
var curHandle C.paddle_master_client

func init() {
	log.Root().SetHandler(
		log.LvlFilterHandler(log.LvlWarn, log.CallerStackHandler("%+v", log.StderrHandler)),
	)
}

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
//
// bufSize is the record buffer size.
func paddle_new_etcd_master_client(etcdEndpoints *C.char, timeout int, bufSize int) C.paddle_master_client {
	p := C.GoString(etcdEndpoints)
	endpoints := strings.Split(p, ",")
	c, err := master.NewClient(
		master.WithEtcd(endpoints, time.Duration(timeout)*time.Second),
		master.WithBuffer(bufSize),
	)
	if err != nil {
		panic(err)
	}

	return add(c)
}

//export paddle_new_master_client
//
// bufSize is the record buffer size.
func paddle_new_master_client(addr *C.char, bufSize int) C.paddle_master_client {
	a := C.GoString(addr)
	c, err := master.NewClient(master.WithAddr(a), master.WithBuffer(bufSize))
	if err != nil {
		panic(err)
	}

	return add(c)
}

//export paddle_release_master_client
func paddle_release_master_client(client C.paddle_master_client) {
	remove(client)
}

//export paddle_start_get_records
func paddle_start_get_records(client C.paddle_master_client, pass C.int) {
	c := get(client)
	c.StartGetRecords(int(pass))
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
		log.Error("error set dataset",
			log.Ctx{"error": err, "paths": paths})
		return C.PADDLE_MASTER_ERROR
	}

	return C.PADDLE_MASTER_OK
}

// paddle_next_record gets the nexts training record.
//
// returns number of bytes of the records if success, -1 if failed, -2 if pass end.
//
//export paddle_next_record
func paddle_next_record(client C.paddle_master_client, record **C.uchar) C.int {
	c := get(client)
	r, err := c.NextRecord()
	if err != nil {
		// NOTE: use errors to indicate pass ends
		if err.Error() == master.ErrAllTaskFailed.Error() ||
			err.Error() == master.ErrNoMoreAvailable.Error() ||
			err.Error() == master.ErrPassBefore.Error() {
			return -2
		}
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

// paddle_request_save_model requests the master server to approve the
// caller to save the model.
//
// returns 1 if the save the model request is approved, 0 if the
// request is rejected because other trainer is saving the model, -1
// if error happened.
//
//export paddle_request_save_model
func paddle_request_save_model(client C.paddle_master_client, trainerID string, blockMS int) C.int {
	c := get(client)
	need, err := c.RequestSaveModel(trainerID, time.Duration(blockMS)*time.Millisecond)
	if err != nil {
		log.Error("error request save model", log.Ctx{"error": err})
		return C.PADDLE_MASTER_ERROR
	}

	if need {
		return C.PADDLE_SAVE_MODEL_OK
	}

	return C.PADDLE_SAVE_MODEL_SKIP
}

//export mem_free
func mem_free(p unsafe.Pointer) {
	// "free" may be a better name for this function, but doing so
	// will cause calling any function of this library from Python
	// ctypes hanging.
	C.free(p)
}

func main() {}
