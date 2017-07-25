// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package master_test

import (
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/PaddlePaddle/Paddle/go/master"
	"github.com/PaddlePaddle/recordio"
	log "github.com/sirupsen/logrus"
)

// tool function for testing output goroutine ids
func goid() int {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	idField := strings.Fields(strings.TrimPrefix(string(buf[:n]), "goroutine "))[0]
	id, err := strconv.Atoi(idField)
	if err != nil {
		panic(fmt.Sprintf("cannot get goroutine id: %v", err))
	}
	return id
}

func TestNextRecord(t *testing.T) {
	log.SetLevel(log.WarnLevel)
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
		s, err := master.NewService(&master.InMemStore{}, 1, time.Second, 1)
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

	w := recordio.NewWriter(f, 1, -1)
	for i := 0; i < total; i++ {
		_, err = w.Write([]byte{byte(i)})
		if err != nil {
			panic(err)
		}
	}

	err = w.Close()
	if err != nil {
		panic(err)
	}

	err = f.Close()
	if err != nil {
		panic(err)
	}

	// start several client to test task fetching
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		// test for multiple concurrent clients
		go func() {
			defer wg.Done()
			// each go-routine needs a single client connection instance
			c, e := master.NewClient(master.WithAddr(fmt.Sprintf(":%d", p)), master.WithBuffer(1))
			if e != nil {
				t.Fatal(e)
			}
			// test for n passes
			for pass := 0; pass < 5; pass++ {
				fmt.Println("starting new pass", pass)
				// wait for last pass end and init for a new pass
				e := c.PassStart()
				if e != nil {
					t.Fatal(e)
				}
				e = c.SetDataset([]string{path})
				if e != nil {
					panic(e)
				}
				received := make(map[byte]bool)
				taskid := 0
				for {
					r, e := c.NextRecord()
					if e != nil {
						// pass done, break for next pass
						if e.Error() == master.ErrAllTaskFinishError.Error() || e.Error() == master.ErrNoMoreAvailableError.Error() {
							break
						}
						t.Fatal(pass, taskid, "Read error:", e)
					}
					if len(r) != 1 {
						t.Fatal(pass, taskid, "Length should be 1.", r)
					}
					if received[r[0]] {
						t.Fatal(pass, taskid, "Received duplicate.", received, r)
					}
					taskid++
					received[r[0]] = true
				}
			}
		}()
	}
	wg.Wait()
}

func TestClientPassSync(t *testing.T) {
	log.SetLevel(log.WarnLevel)
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
		s, err := master.NewService(&master.InMemStore{}, 1, time.Second, 1)
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

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			curAddr := make(chan string, 1)
			curAddr <- fmt.Sprintf(":%d", p)
			c, e := master.NewClient(master.WithAddr(fmt.Sprintf(":%d", p)), master.WithBuffer(10))
			if e != nil {
				t.Fatal(e)
			}
			for pass := 0; pass < 10; pass++ {
				e := c.PassStart()
				if e != nil {
					t.Fatal(e)
				}
				// simulate run pass
				time.Sleep(time.Duration(rand.Intn(3)) * time.Second)
				e = c.PassFinish()
				if e != nil {
					t.Fatal(e)
				}
			}
		}()
	}
	wg.Wait()
}
