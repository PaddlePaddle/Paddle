package master_test

import (
	"fmt"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/PaddlePaddle/Paddle/go/master"
	"github.com/PaddlePaddle/recordio"
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
		s := master.NewService(chunkPerTask, time.Second, 1)
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
	const p = "/tmp/master_client_test_0"
	f, err := os.Create(p)
	if err != nil {
		panic(err)
	}

	for i := 0; i < totalTask*chunkPerTask; i++ {
		w := recordio.NewWriter(f, -1, -1)
		w.Write(nil)
		// call Close to force RecordIO writing a chunk.
		w.Close()
	}
	f.Close()

	c := master.NewClient(addresser(fmt.Sprintf(":%d", port)))
	c.SetDataset([]string{p})

	checkOnePass := func(i int) {
		var tasks []master.Task
		for i := 0; i < totalTask; i++ {
			task, err := c.GetTask()
			if err != nil {
				t.Fatal(i, err)
			}
			tasks = append(tasks, task)
		}

		_, err = c.GetTask()
		if err == nil {
			t.Fatal(i, "should get error.")
		}

		for _, task := range tasks {
			err = c.TaskFinished(task.ID)
			if err != nil {
				t.Fatal(i, err)
			}
		}
	}

	for i := 0; i < 10; i++ {
		checkOnePass(i)
	}
}
