// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package client_test

import (
	"context"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	"github.com/PaddlePaddle/Paddle/go/pserver/client"
	"github.com/coreos/etcd/clientv3"
	log "github.com/inconshreveable/log15"
)

const (
	numPserver    = 10
	etcdEndpoints = "127.0.0.1:2379"
	timeout       = 2 * time.Second
)

var pserverClientPorts [numPserver]int

// this function init pserver client and return their ports in an array.
func initClient() [numPserver]int {
	var ports [numPserver]int
	for i := 0; i < numPserver; i++ {
		l, err := net.Listen("tcp", ":0")
		if err != nil {
			panic(err)
		}

		ss := strings.Split(l.Addr().String(), ":")
		p, err := strconv.Atoi(ss[len(ss)-1])
		if err != nil {
			panic(err)
		}
		ports[i] = p

		go func(l net.Listener) {
			var cp pserver.Checkpoint
			s, err := pserver.NewService(0, time.Hour, "", nil, cp)
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
	}
	return ports
}

func initNativeClient() {
	pserverClientPorts = initClient()
}

func initEtcdClient() {
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{etcdEndpoints},
		DialTimeout: time.Second * time.Duration(1),
	})
	if err != nil {
		log.Error("error init etcd client", log.Ctx{"error": err})
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	_, err = client.Delete(ctx, pserver.PsDesired)
	if err != nil {
		panic(err)
	}

	_, err = client.Delete(ctx, pserver.PsPath)
	if err != nil {
		panic(err)
	}

	_, err = client.Put(ctx, pserver.PsDesired, strconv.Itoa(numPserver))
	if err != nil {
		panic(err)
	}

	ports := initClient()
	for i := 0; i < numPserver; i++ {
		_, err = client.Put(ctx, pserver.PsPath+strconv.Itoa(i), ":"+strconv.Itoa(ports[i]))
		if err != nil {
			panic(err)
		}
	}
	cancel()
	err = client.Close()
	if err != nil {
		panic(err)
	}
}

type selector bool

func (s selector) Select() (bool, error) {
	return bool(s), nil
}

func (s selector) Done() error {
	return nil
}

type lister []client.Server

func (l lister) List() []client.Server {
	return l
}

func testClient(t *testing.T, c *client.Client) {
	selected, err := c.BeginInitParams()
	if err != nil {
		t.Fatal(err)
	}

	if !selected {
		t.Fatal("should be selected.")
	}

	const numParameter = 1000
	config, err := ioutil.ReadFile("./c/test/testdata/optimizer.pb")
	if err != nil {
		t.Fatalf("read optimizer proto failed")
	}

	var wg sync.WaitGroup
	for i := 0; i < numParameter; i++ {
		wg.Add(1)
		go func(i int) {
			var p pserver.Parameter
			p.Name = "p_" + strconv.Itoa(i)
			p.ElementType = pserver.Float32
			p.Content = make([]byte, (i+1)*100)
			err := c.InitParam(pserver.ParameterWithConfig{Param: p, Config: config})
			if err != nil {
				t.Fatal(err)
			}
			wg.Done()
		}(i)
	}
	wg.Wait()

	err = c.FinishInitParams()
	if err != nil {
		t.Fatal(err)
	}

	var grads []pserver.Gradient
	for i := 0; i < numParameter; i++ {
		var g pserver.Gradient
		g.Name = "p_" + strconv.Itoa(i)
		g.ElementType = pserver.Float32
		g.Content = make([]byte, (i+1)*100)
		grads = append(grads, g)
	}

	const paramPerGroup = 10
	const numGroups = numParameter / paramPerGroup

	// shuffle send grads order
	for i := range grads {
		j := rand.Intn(i + 1)
		grads[i], grads[j] = grads[j], grads[i]
	}

	for i := 0; i < numGroups; i++ {
		var gs []pserver.Gradient
		if i == numGroups-1 {
			gs = grads[i*paramPerGroup:]
		} else {
			gs = grads[i*paramPerGroup : (i+1)*paramPerGroup]
		}

		wg.Add(1)
		go func(gs []pserver.Gradient) {
			err := c.SendGrads(gs)
			if err != nil {
				t.Fatal(err)
			}
			wg.Done()
		}(gs)
	}

	names := make([]string, numParameter)
	for i := 0; i < numParameter; i++ {
		names[i] = "p_" + strconv.Itoa(i)
	}

	for i := 0; i < numGroups; i++ {
		var ns []string
		if i == numGroups-1 {
			ns = names[i*paramPerGroup:]
		} else {
			ns = names[i*paramPerGroup : (i+1)*paramPerGroup]
		}

		wg.Add(1)
		go func(ns []string) {
			params, err := c.GetParams(ns)
			if err != nil {
				t.Fatal(err)
			}

			if len(ns) != len(params) {
				t.Fatalf("parameter size not match, need: %d, have: %d", len(names), len(params))
			}

			for i := range params {
				if ns[i] != params[i].Name {
					t.Fatalf("order of returned parameter does not required: parameter name: %s, required name: %s", ns[i], params[i].Name)
				}
			}
			wg.Done()
		}(ns)
	}

	wg.Wait()
}

func TestNativeClient(t *testing.T) {
	initNativeClient()
	servers := make([]client.Server, numPserver)
	for i := 0; i < numPserver; i++ {
		servers[i] = client.Server{Index: i, Addr: ":" + strconv.Itoa(pserverClientPorts[i])}
	}
	c1 := client.NewClient(lister(servers), len(servers), selector(true))
	testClient(t, c1)
}

// EtcdClient is a disabled test, since we have not embedded etcd into
// our test.
func EtcdClient(t *testing.T) {
	initEtcdClient()
	etcdClient := client.NewEtcd(etcdEndpoints)
	c2 := client.NewClient(etcdClient, etcdClient.Desired(), selector(true))
	testClient(t, c2)
}
