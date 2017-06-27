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
	"github.com/stretchr/testify/assert"
)

type testDB struct{}

func (testDB) WaitMasterReady(key string, interval int) []byte {
	return []byte("localhost:12345")
}
func (testDB) WatchWithKey(key string, valCh chan string) {
	for i := 1; i < 100; i++ {
		valCh <- fmt.Sprintf("localhost:%d", 12345+i)
		time.Sleep(time.Second * time.Duration(1))
	}
}

func TestNextRecord(t *testing.T) {
	const (
		path  = "/tmp/master_client_TestFull"
		total = 50
	)
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
		s, err := master.NewService(&master.InMemStore{}, 10, time.Second, 1)
		if err != nil {
			panic(err)
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

	w := recordio.NewWriter(f, -1, -1)
	for i := 0; i < total; i++ {
		w.Write([]byte{byte(i)})
	}
	w.Close()
	f.Close()
	curAddr := make(chan string)
	c := master.NewClient(curAddr, 10)
	curAddr <- fmt.Sprintf(":%d", p)
	c.SetDataset([]string{path})
	for pass := 0; pass < 50; pass++ {
		received := make(map[byte]bool)
		for i := 0; i < total; i++ {
			r := c.NextRecord()
			if len(r) != 1 {
				t.Fatal("Length should be 1.", r)
			}
			if received[r[0]] {
				t.Fatal("Received duplicate.", received, r)
			}
			received[r[0]] = true
		}
	}
}

func TestNewEtcdMasterClient(t *testing.T) {
	db := testDB{}
	c := master.NewEtcdMasterClient(db, 3)
	assert.NotNil(t, c)
}
