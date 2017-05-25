package main

import (
	"flag"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/PaddlePaddle/Paddle/paddle/go/master"
	"github.com/wangkuiyi/recordio"
)

const (
	taskTimeoutDur = 20 * time.Minute
	taskTimeoutMax = 3
)

func main() {
	port := flag.Int("p", 0, "port of the master server")
	dataset := flag.String("d", "", "dataset: comma separated path to RecordIO files")
	faultTolerant := flag.Bool("fault-tolerance", false, "enable fault tolerance (requires etcd).")
	flag.Parse()

	if *dataset == "" {
		panic("no dataset specified.")
	}

	if *faultTolerant {
		panic("fault tolernat not implemented.")
	}

	var chunks []master.Chunk
	paths := strings.Split(*dataset, ",")
	idx := 0
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			panic(err)
		}

		index, err := recordio.LoadIndex(f)
		if err != nil {
			panic(err)
		}
		f.Close()

		count := index.NumChunks()
		for i := 0; i < count; i++ {
			chunk := master.Chunk{
				Idx:   idx,
				Path:  path,
				Index: *index.ChunkIndex(i),
			}
			chunks = append(chunks, chunk)
		}
	}

	s := master.NewService(chunks, taskTimeoutDur, taskTimeoutMax)
	err := rpc.Register(s)
	if err != nil {
		panic(err)
	}

	rpc.HandleHTTP()
	l, err := net.Listen("tcp", ":"+strconv.Itoa(*port))
	if err != nil {
		panic(err)
	}

	err = http.Serve(l, nil)
	if err != nil {
		panic(err)
	}
}
