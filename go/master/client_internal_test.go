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
		s, e := NewService(&InMemStore{}, chunkPerTask, time.Second, 1)
		if err != nil {
			panic(e)
		}

		server := rpc.NewServer()
		err = server.Register(s)
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
		_, err = w.Write(nil)
		if err != nil {
			panic(err)
		}

		// call Close to force RecordIO writing a chunk.
		err = w.Close()
		if err != nil {
			panic(err)
		}
	}
	err = f.Close()
	if err != nil {
		panic(err)
	}

	// Manually intialize client to avoid calling c.getRecords()
	c := &Client{}
	c.conn = connection.New()
	addr := fmt.Sprintf(":%d", p)
	ch := make(chan string, 1)
	ch <- addr
	go c.monitorMaster(ch)
	req := SetDatasetRequest{}
	req.GlobPaths = []string{path}
	req.NumPasses = 10
	err = c.SetDataset(req)
	if err != nil {
		t.Fatal(err)
	}
	checkOnePass := func(i int) {
		var tasks []Task
		for idx := 0; idx < totalTask; idx++ {
			task, e := c.getTask()
			if e != nil {
				t.Fatalf("Error: %v, pass: %d\n", e, i)
			}
			tasks = append(tasks, task)
		}
		_, err = c.getTask()
		if err == nil {
			t.Fatalf("Should get error, pass: %d\n", i)
		}

		err = c.taskFinished(tasks[0].Meta.ID)
		if err != nil {
			t.Fatalf("Error: %v, pass: %d\n", err, i)
		}

		err = c.taskFailed(tasks[0].Meta)
		if err != nil {
			t.Fatalf("Error: %v, pass: %d\n", err, i)
		}

		tasks = tasks[1:]
		task, e := c.getTask()
		if e.Error() != "no more available task" {
			t.Fatal(e)
		}
		tasks = append(tasks, task)

		for _, task := range tasks {
			err = c.taskFinished(task.Meta.ID)
			if err != nil {
				t.Fatalf("Error: %v, pass: %d\n", err, i)
			}
		}
	}

	for i := 0; i < req.NumPasses-1; i++ {
		checkOnePass(i)
	}
	// last pass check all task finish of all passes
	for idx := 0; idx < totalTask; idx++ {
		task, e := c.getTask()
		if e != nil {
			t.Fatalf("Error: %v\n", e)
		}
		err = c.taskFinished(task.Meta.ID)
		if idx < totalTask-1 {
			if err != nil {
				t.Fatal(err)
			}
		} else {
			// FIXME: use error string to identify error
			if err.Error() != "all task done" {
				t.Fatal(err)
			}
		}
	}
	_, e := c.getTask()
	if e == nil || e.Error() != "all task done" {
		t.Error(e)
	}
}
