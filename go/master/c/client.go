package main

/*

typedef int paddle_master_client;
*/
import "C"

import (
	"log"
	"sync"
	"unsafe"

	"github.com/PaddlePaddle/Paddle/go/master"
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

type addresser string

func (a addresser) Address() string {
	return string(a)
}

//paddle_new_master_client
func paddle_new_master_client(addr *C.char, buf_size C.int) C.paddle_master_client {
	a := C.GoString(addr)
	c := master.NewClient(addresser(a), int(buf_size))
	return add(c)
}

//export paddle_new_etcd_master_client
func paddle_new_etcd_master_client(etcd_addr *C.char) C.paddle_master_client {
	// TODO(helin): fault tolerant master client using etcd.
	panic("not implemented.")
}

//export paddle_set_dataset
func paddle_set_dataset(client C.paddle_master_client, path **C.char, size C.int) C.int {
	c := get(client)
	var paths []string
	for i := 0; i < int(size); i++ {
		ptr := (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(path)) + uintptr(size)))
		str := C.GoString(*ptr)
		paths = append(paths, str)
	}
	err := c.SetDataset(paths)
	if err != nil {
		log.Println(err)
		return -1
	}

	return 0
}

func main() {}
