package main

import (
	"fmt"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/namsral/flag"

	"github.com/PaddlePaddle/Paddle/paddle/go/master"
	"github.com/PaddlePaddle/Paddle/paddle/go/recordio"
)

func main() {
	port := flag.Int("port", 8080, "port of the master server.")
	dataset := flag.String("training_dataset", "", "dataset: comma separated path to RecordIO paths, supports golb patterns.")
	faultTolerance := flag.Bool("fault_tolerance", false, "enable fault tolerance (requires etcd).")
	taskTimeoutDur := flag.Duration("task_timout_dur", 20*time.Minute, "task timout duration.")
	taskTimeoutMax := flag.Int("task_timeout_max", 3, "max timtout count for each task before it being declared failed task.")
	chunkPerTask := flag.Int("chunk_per_task", 10, "chunk per task.")
	flag.Parse()

	if *dataset == "" {
		panic("no dataset specified.")
	}

	if *faultTolerance {
		panic("fault tolernance not implemented.")
	}

	var chunks []master.Chunk
	var paths []string
	ss := strings.Split(*dataset, ",")
	fmt.Println(ss)
	for _, s := range ss {
		match, err := filepath.Glob(s)
		if err != nil {
			panic(err)
		}
		paths = append(paths, match...)
	}

	if len(paths) == 0 {
		panic("no valid datset specified.")
	}

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

	s := master.NewService(chunks, *chunkPerTask, *taskTimeoutDur, *taskTimeoutMax)
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
