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
	"sync"
	"unsafe"

	"github.com/PaddlePaddle/Paddle/go/master"
	log "github.com/sirupsen/logrus"
)

var nullPtr = unsafe.Pointer(uintptr(0))
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
	e, err := master.NewEtcdClient(p, timeout)
	if err != nil {
		panic(err)
	}
	c := master.NewEtcdMasterClient(e, bufSize)
	return add(c)
}

//export paddle_new_master_client
func paddle_new_master_client(addr *C.char, bufSize int) C.paddle_master_client {
	a := C.GoString(addr)
	ch := make(chan string)
	c := master.NewClient(ch, bufSize)
	ch <- a
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

//export paddle_next_record
func paddle_next_record(client C.paddle_master_client, record **C.uchar) C.int {
	c := get(client)
	r := c.NextRecord()
	if len(r) == 0 {
		*record = (*C.uchar)(nullPtr)
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
