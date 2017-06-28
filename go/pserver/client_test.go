package pserver_test

import (
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"strings"
	"testing"

	"github.com/PaddlePaddle/Paddle/go/pserver"
)

const numPserver = 10

var port [numPserver]int

func init() {
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
		port[i] = p

		go func(l net.Listener) {
			s, err := pserver.NewService(0)
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
}

type selector bool

func (s selector) Select() bool {
	return bool(s)
}

type lister []pserver.Server

func (l lister) List() []pserver.Server {
	return l
}

func TestClientFull(t *testing.T) {
	servers := make([]pserver.Server, numPserver)
	for i := 0; i < numPserver; i++ {
		servers[i] = pserver.Server{Index: i, Addr: ":" + strconv.Itoa(port[i])}
	}
	c := pserver.NewClient(lister(servers), len(servers), selector(true))
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
