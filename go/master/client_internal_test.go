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
		s, sErr := NewService(&InMemStore{}, chunkPerTask, time.Second, 1)
		if sErr != nil {
			panic(sErr)
		}

		server := rpc.NewServer()
		sErr = server.Register(s)
		if sErr != nil {
			panic(sErr)
		}

		mux := http.NewServeMux()
		mux.Handle(rpc.DefaultRPCPath, server)
		sErr = http.Serve(l, mux)
		if sErr != nil {
			panic(sErr)
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

	checkOnePass := func(i int) {
		var tasks []Task
		for idx := 0; idx < totalTask; idx++ {
			task, cErr := c.getTask()
			if cErr != nil && cErr.Error() != ErrNoMoreAvailableError.Error() {
				t.Fatalf("error: %v, pass: %d\n", cErr, i)
			}
			tasks = append(tasks, task)
		}

		// getting task before task finishes should return error
		_, cErr := c.getTask()
		if cErr == nil {
			t.Fatalf("Should get error, pass: %d\n", i)
		}

		cErr = c.taskFinished(tasks[0].Meta.ID)
		if cErr != nil {
			t.Fatalf("Error: %v, pass: %d\n", cErr, i)
		}
		// call taskFailed once won't put the task to failed queue, just ensure
		// the call
		cErr = c.taskFailed(tasks[0].Meta)
		if cErr != nil {
			t.Fatalf("Error: %v, pass: %d\n", cErr, i)
		}

		tasks = tasks[1:]
		_, cErr = c.getTask()
		if cErr != nil && cErr.Error() != ErrNoMoreAvailableError.Error() && cErr.Error() != ErrAllTaskFinishError.Error() {
			t.Fatalf("Should be ErrNoMoreAvailableError or ErrAllTaskFinishError: %s", cErr)
		}

		for _, task := range tasks {
			cErr = c.taskFinished(task.Meta.ID)
			if cErr != nil && cErr.Error() != ErrAllTaskFinishError.Error() {
				t.Fatalf("Non-ErrAllTaskFinishError: %v, pass: %d\n", cErr, i)
			}
		}
	}

	for i := 0; i < 10; i++ {
		// init pass data
		err = c.SetDataset([]string{path})
		if err != nil {
			panic(err)
		}
		checkOnePass(i)
	}
}
