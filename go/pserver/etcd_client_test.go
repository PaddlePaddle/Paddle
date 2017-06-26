package pserver_test

import (
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"strings"
	"testing"
	"time"
	"context"

	"github.com/coreos/etcd/clientv3"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	log "github.com/sirupsen/logrus"
)

const (
	numPserver = 10
	defaultEtcdAddr = "127.0.0.1:2379"
	timeout = time.Second * time.Duration(2)
)

func init() {
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{defaultEtcdAddr},
		DialTimeout: time.Second * time.Duration(1),
	})
	if err != nil {
		log.Errorf("err %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	client.Put(ctx, pserver.DefaultPsDesiredPath, strconv.Itoa(numPserver))

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
		client.Put(ctx, pserver.DefaultPsBasePath + strconv.Itoa(i), ":" + strconv.Itoa(p))

		go func(l net.Listener) {
			s, err := pserver.NewService("", time.Second*5)
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
	cancel()
	client.Close()
}

type selector bool

func (s selector) Select() bool {
	return bool(s)
}

func TestEtcdClientFull(t *testing.T) {
	lister, psDesired := pserver.NewEtcdAddrLister(defaultEtcdAddr)
	c := pserver.NewClient(lister, psDesired, selector(true))
	selected := c.BeginInitParams()
	if !selected {
		t.Fatal("should be selected.")
	}

	const numParameter = 100
	for i := 0; i < numParameter; i++ {
		var p pserver.Parameter
		p.Name = "p_" + strconv.Itoa(i)
		p.ElementType = pserver.Float32
		p.Content = make([]byte, (i+1)*100)
		err := c.InitParam(pserver.ParameterWithConfig{Param: p})
		if err != nil {
			t.Fatal(err)
		}
	}

	err := c.FinishInitParams()
	if err != nil {
		t.Fatal(err)
	}

	var grads []pserver.Gradient
	for i := 0; i < numParameter/2; i++ {
		var g pserver.Gradient
		g.Name = "p_" + strconv.Itoa(i)
		g.ElementType = pserver.Float32
		g.Content = make([]byte, (i+1)*100)
		grads = append(grads, g)
	}

	err = c.SendGrads(grads)
	if err != nil {
		t.Fatal(err)
	}

	names := make([]string, numParameter)
	for i := 0; i < numParameter; i++ {
		names[i] = "p_" + strconv.Itoa(i)
	}

	params, err := c.GetParams(names)
	if err != nil {
		t.Fatal(err)
	}

	if len(names) != len(params) {
		t.Fatalf("parameter size not match, need: %d, have: %d", len(names), len(params))
	}

	for i := range params {
		if names[i] != params[i].Name {
			t.Fatalf("order of returned parameter does not required: parameter name: %s, required name: %s", names[i], params[i].Name)
		}
	}
}
