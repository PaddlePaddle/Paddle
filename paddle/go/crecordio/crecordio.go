package main

/*
#include <string.h>

typedef int reader;
typedef int writer;
*/
import "C"

import (
	"io"
	"log"
	"os"
	"path/filepath"
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
	buffer chan []byte
	cancel chan struct{}
}

func read(paths []string, buffer chan<- []byte, cancel chan struct{}) {
	var curFile *os.File
	var curScanner *recordio.Scanner
	var pathIdx int

	var nextFile func() bool
	nextFile = func() bool {
		if pathIdx >= len(paths) {
			return false
		}

		path := paths[pathIdx]
		pathIdx++
		f, err := os.Open(path)
		if err != nil {
			return nextFile()
		}

		idx, err := recordio.LoadIndex(f)
		if err != nil {
			log.Println(err)
			err = f.Close()
			if err != nil {
				log.Println(err)
			}

			return nextFile()
		}

		curFile = f
		curScanner = recordio.NewScanner(f, idx, 0, -1)
		return true
	}

	more := nextFile()
	if !more {
		close(buffer)
		return
	}

	closeFile := func() {
		err := curFile.Close()
		if err != nil {
			log.Println(err)
		}
		curFile = nil
	}

	for {
		for curScanner.Scan() {
			select {
			case buffer <- curScanner.Record():
			case <-cancel:
				close(buffer)
				closeFile()
				return
			}
		}

		if err := curScanner.Error(); err != nil && err != io.EOF {
			log.Println(err)
		}

		closeFile()
		more := nextFile()
		if !more {
			close(buffer)
			return
		}
	}
}

//export paddle_new_writer
func paddle_new_writer(path *C.char) C.writer {
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

func cArrayToSlice(p unsafe.Pointer, len int) []byte {
	if p == nullPtr {
		return nil
	}

	// create a Go clice backed by a C array, reference:
	// https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	//
	// Go garbage collector will not interact with this data, need
	// to be freed from C side.
	return (*[1 << 30]byte)(p)[:len:len]
}

//export paddle_writer_write
func paddle_writer_write(writer C.writer, buf *C.uchar, size C.int) int {
	w := getWriter(writer)
	b := cArrayToSlice(unsafe.Pointer(buf), int(size))
	_, err := w.w.Write(b)
	if err != nil {
		log.Println(err)
		return -1
	}

	return 0
}

//export paddle_writer_release
func paddle_writer_release(writer C.writer) {
	w := removeWriter(writer)
	w.w.Close()
	w.f.Close()
}

//export paddle_new_reader
func paddle_new_reader(path *C.char, bufferSize C.int) C.reader {
	p := C.GoString(path)
	ss := strings.Split(p, ",")
	var paths []string
	for _, s := range ss {
		match, err := filepath.Glob(s)
		if err != nil {
			log.Printf("error applying glob to %s: %v\n", s, err)
			return -1
		}

		paths = append(paths, match...)
	}

	if len(paths) == 0 {
		log.Println("no valid path provided.", p)
		return -1
	}

	buffer := make(chan []byte, int(bufferSize))
	cancel := make(chan struct{})
	r := &reader{buffer: buffer, cancel: cancel}
	go read(paths, buffer, cancel)
	return addReader(r)
}

//export paddle_reader_next_item
func paddle_reader_next_item(reader C.reader, size *C.int) *C.uchar {
	r := getReader(reader)
	buf, ok := <-r.buffer
	if !ok {
		// channel closed and empty, reached EOF.
		*size = -1
		return (*C.uchar)(nullPtr)
	}

	if len(buf) == 0 {
		// empty item
		*size = 0
		return (*C.uchar)(nullPtr)
	}

	ptr := C.malloc(C.size_t(len(buf)))
	C.memcpy(ptr, unsafe.Pointer(&buf[0]), C.size_t(len(buf)))
	*size = C.int(len(buf))
	return (*C.uchar)(ptr)
}

//export paddle_reader_release
func paddle_reader_release(reader C.reader) {
	r := removeReader(reader)
	close(r.cancel)
}

func main() {} // Required but ignored
