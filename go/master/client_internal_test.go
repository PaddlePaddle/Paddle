package master

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

	log "github.com/sirupsen/logrus"

	"github.com/PaddlePaddle/Paddle/go/connection"
	"github.com/PaddlePaddle/recordio"
)

const (
	totalTask    = 20
	chunkPerTask = 10
)

func init() {
	log.SetLevel(log.ErrorLevel)
}

type TestAddresser string

func (a TestAddresser) Address() string {
	return string(a)
}

func TestGetFinishTask(t *testing.T) {
	const path = "/tmp/master_client_test_0"

	l, err := net.Listen("tcp", ":0")
	if err != nil {
		panic(err)
	}

	ss := strings.Split(l.Addr().String(), ":")
	p, err := strconv.Atoi(ss[len(ss)-1])
	if err != nil {
		panic(err)
	}

	go func(l net.Listener) {
		s := NewService(chunkPerTask, time.Second, 1)
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

	f, err := os.Create(path)
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

	c := &Client{}
	c.conn = connection.New()
	go c.monitorMaster(TestAddresser(fmt.Sprintf(":%d", p)))
	c.SetDataset([]string{path})

	checkOnePass := func(i int) {
		var tasks []Task
		for idx := 0; idx < totalTask; idx++ {
			task, err := c.getTask()
			if err != nil {
				t.Fatal(err, " pass:", i)
			}
			tasks = append(tasks, task)
		}

		_, err = c.getTask()
		if err == nil {
			t.Fatal("Should get error. Pass:", i)
		}

		err = c.taskFinished(tasks[0].ID)
		if err != nil {
			t.Fatal(err, "pass:", i)
		}
		tasks = tasks[1:]
		task, err := c.getTask()
		if err != nil {
			t.Fatal(err)
		}
		tasks = append(tasks, task)

		for _, task := range tasks {
			err = c.taskFinished(task.ID)
			if err != nil {
				t.Fatal(err, " pass:", i)
			}
		}
	}

	for i := 0; i < 10; i++ {
		checkOnePass(i)
	}
}
