package client_test

import (
	"context"
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	"github.com/PaddlePaddle/Paddle/go/pserver/client"
	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
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
		log.Errorf("err %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	client.Delete(ctx, pserver.PsDesired)
	client.Delete(ctx, pserver.PsPath)
	client.Put(ctx, pserver.PsDesired, strconv.Itoa(numPserver))
	ports := initClient()
	for i := 0; i < numPserver; i++ {
		client.Put(ctx, pserver.PsPath+strconv.Itoa(i), ":"+strconv.Itoa(ports[i]))
	}
	cancel()
	client.Close()
}

type selector bool

func (s selector) Select() bool {
	return bool(s)
}

type lister []client.Server

func (l lister) List() []client.Server {
	return l
}

func ClientTest(t *testing.T, c *client.Client) {
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

func TestNativeClient(t *testing.T) {
	initNativeClient()
	servers := make([]client.Server, numPserver)
	for i := 0; i < numPserver; i++ {
		servers[i] = client.Server{Index: i, Addr: ":" + strconv.Itoa(pserverClientPorts[i])}
	}
	c1 := client.NewClient(lister(servers), len(servers), selector(true))
	ClientTest(t, c1)
}

//TODO(Qiao: tmperary disable etcdClient test for dependency of etcd)
func EtcdClient(t *testing.T) {
	initEtcdClient()
	etcd_client, _ := client.NewEtcd(etcdEndpoints)
	c2 := client.NewClient(etcd_client, etcd_client.Desired(), selector(true))
	ClientTest(t, c2)
}
