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

	curAddr := make(chan string, 1)
	curAddr <- fmt.Sprintf(":%d", p)
	c := master.NewClient(curAddr, 10)
	err = c.SetDataset([]string{path})
	if err != nil {
		panic(err)
	}

	for pass := 0; pass < 50; pass++ {
		received := make(map[byte]bool)
		for i := 0; i < total; i++ {
			r, err := c.NextRecord()
			if err != nil {
				t.Fatal(pass, i, "Read error:", err)
			}

			if len(r) != 1 {
				t.Fatal(pass, i, "Length should be 1.", r)
			}

			if received[r[0]] {
				t.Fatal(pass, i, "Received duplicate.", received, r)
			}
			received[r[0]] = true
		}
	}
}
