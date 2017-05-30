package main

/*
typedef int reader;
typedef int writer;
*/
import "C"

import "sync"

var mu sync.Mutex
var handleMap = make(map[C.reader]*reader)
var curHandle C.reader
var writerMap = make(map[C.writer]*writer)
var curWriterHandle C.writer

func addReader(r *reader) C.reader {
	mu.Lock()
	defer mu.Unlock()
	reader := curHandle
	curHandle++
	handleMap[reader] = r
	return reader
}

func getReader(reader C.reader) *reader {
	mu.Lock()
	defer mu.Unlock()
	return handleMap[reader]
}

func removeReader(reader C.reader) *reader {
	mu.Lock()
	defer mu.Unlock()
	r := handleMap[reader]
	delete(handleMap, reader)
	return r
}

func addWriter(w *writer) C.writer {
	mu.Lock()
	defer mu.Unlock()
	writer := curWriterHandle
	curWriterHandle++
	writerMap[writer] = w
	return writer
}

func getWriter(writer C.writer) *writer {
	mu.Lock()
	defer mu.Unlock()
	return writerMap[writer]
}

func removeWriter(writer C.writer) *writer {
	mu.Lock()
	defer mu.Unlock()
	w := writerMap[writer]
	delete(writerMap, writer)
	return w
}
