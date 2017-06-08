package master_test

import (
	"fmt"
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/PaddlePaddle/Paddle/go/master"
)

const (
	totalTask    = 20
	chunkPerTask = 10
)

var port int

func init() {
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		panic(err)
	}

	ss := strings.Split(l.Addr().String(), ":")
	p, err := strconv.Atoi(ss[len(ss)-1])
	if err != nil {
		panic(err)
	}
	port = p

	go func(l net.Listener) {
		chunks := make([]master.Chunk, totalTask)
		s := master.NewService(chunks, chunkPerTask, time.Second, 1)
		server := rpc.NewServer()
		err := server.Register(s)
		if err != nil {
			panic(err)
		}

		mux := http.NewServeMux()
		mux.Handle(rpc.DefaultRPCPath, server)
		err = http.Serve(l, mux)
		if err != nil {
			panic(err)
		}
	}(l)
}

type addresser string

func (a addresser) Address() string {
	return string(a)
}

func TestClientFull(t *testing.T) {
	c := master.NewClient(addresser(fmt.Sprintf(":%d", port)))

	for i := 0; i < 5*totalTask/chunkPerTask; i++ {
		task, err := c.GetTask()
		if err != nil {
			panic(err)
		}

		if len(task.Chunks) != chunkPerTask {
			t.Fatal("wrong number of chunk per task", len(task.Chunks))
		}

		err = c.TaskFinished(task.ID)
		if err != nil {
			panic(err)
		}
	}
}
