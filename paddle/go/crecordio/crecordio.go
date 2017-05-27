package main

/*
#include <string.h>

typedef int reader;
typedef int writer;
*/
import "C"

import (
	"log"
	"os"
	"strings"
	"unsafe"

	"github.com/PaddlePaddle/Paddle/paddle/go/recordio"
)

var nullPtr = unsafe.Pointer(uintptr(0))

type writer struct {
	w *recordio.Writer
	f *os.File
}

type reader struct {
	scanner *recordio.MultiScanner
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

//export create_recordio_writer
func create_recordio_writer(path *C.char) C.writer {
	p := C.GoString(path)
	f, err := os.Create(p)
	if err != nil {
		log.Println(err)
		return -1
	}

	w := recordio.NewWriter(f, -1, -1)
	writer := &writer{f: f, w: w}
	return addWriter(writer)
}

//export write_recordio
func write_recordio(writer C.writer, buf *C.uchar, size C.int) int {
	w := getWriter(writer)
	b := cArrayToSlice(unsafe.Pointer(buf), int(size))
	_, err := w.w.Write(b)
	if err != nil {
		log.Println(err)
		return -1
	}

	return 0
}

//export release_recordio
func release_recordio(writer C.writer) {
	w := removeWriter(writer)
	w.w.Close()
	w.f.Close()
}

//export create_recordio_reader
func create_recordio_reader(path *C.char) C.reader {
	p := C.GoString(path)
	s, err := recordio.NewMultiScanner(strings.Split(p, ","))
	if err != nil {
		log.Println(err)
		return -1
	}

	r := &reader{scanner: s}
	return addReader(r)
}

//export read_next_item
func read_next_item(reader C.reader, size *C.int) *C.uchar {
	r := getReader(reader)
	if r.scanner.Scan() {
		buf := r.scanner.Record()
		*size = C.int(len(buf))

		if len(buf) == 0 {
			return (*C.uchar)(nullPtr)
		}

		ptr := C.malloc(C.size_t(len(buf)))
		C.memcpy(ptr, unsafe.Pointer(&buf[0]), C.size_t(len(buf)))
		return (*C.uchar)(ptr)
	}

	*size = -1
	return (*C.uchar)(nullPtr)
}

//export release_recordio_reader
func release_recordio_reader(reader C.reader) {
	r := removeReader(reader)
	r.scanner.Close()
}

func main() {} // Required but ignored
