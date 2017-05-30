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

	"github.com/PaddlePaddle/Paddle/go/recordio"
)

var nullPtr = unsafe.Pointer(uintptr(0))

type writer struct {
	w *recordio.Writer
	f *os.File
}

type reader struct {
	scanner *recordio.Scanner
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

//export recordio_write
func recordio_write(writer C.writer, buf *C.uchar, size C.int) C.int {
	w := getWriter(writer)
	b := cArrayToSlice(unsafe.Pointer(buf), int(size))
	c, err := w.w.Write(b)
	if err != nil {
		log.Println(err)
		return -1
	}

	return C.int(c)
}

//export release_recordio_writer
func release_recordio_writer(writer C.writer) {
	w := removeWriter(writer)
	w.w.Close()
	w.f.Close()
}

//export create_recordio_reader
func create_recordio_reader(path *C.char) C.reader {
	p := C.GoString(path)
	s, err := recordio.NewScanner(strings.Split(p, ",")...)
	if err != nil {
		log.Println(err)
		return -1
	}

	r := &reader{scanner: s}
	return addReader(r)
}

//export recordio_read
func recordio_read(reader C.reader, record **C.uchar) C.int {
	r := getReader(reader)
	if r.scanner.Scan() {
		buf := r.scanner.Record()
		if len(buf) == 0 {
			*record = (*C.uchar)(nullPtr)
			return 0
		}

		size := C.int(len(buf))
		*record = (*C.uchar)(C.malloc(C.size_t(len(buf))))
		C.memcpy(unsafe.Pointer(*record), unsafe.Pointer(&buf[0]), C.size_t(len(buf)))
		return size
	}

	return -1
}

//export release_recordio_reader
func release_recordio_reader(reader C.reader) {
	r := removeReader(reader)
	r.scanner.Close()
}

func main() {} // Required but ignored
