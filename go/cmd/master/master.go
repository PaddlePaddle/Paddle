package main

import (
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"time"

	"github.com/namsral/flag"

	"github.com/PaddlePaddle/Paddle/go/master"
)

func main() {
	port := flag.Int("port", 8080, "port of the master server.")

	faultTolerance := flag.Bool("fault_tolerance", false, "enable fault tolerance (requires etcd).")
	taskTimeoutDur := flag.Duration("task_timout_dur", 20*time.Minute, "task timout duration.")
	taskTimeoutMax := flag.Int("task_timeout_max", 3, "max timtout count for each task before it being declared failed task.")
	chunkPerTask := flag.Int("chunk_per_task", 10, "chunk per task.")
	flag.Parse()

	if *faultTolerance {
		panic("fault tolernance not implemented.")

	}

	s := master.NewService(*chunkPerTask, *taskTimeoutDur, *taskTimeoutMax)
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
