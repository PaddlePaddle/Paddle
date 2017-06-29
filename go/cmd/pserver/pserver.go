package main

import (
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"time"

	"github.com/namsral/flag"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	log "github.com/sirupsen/logrus"
)

func main() {
	port := flag.Int("port", 0, "port of the pserver")
	etcdEndpoint := flag.String("etcd-endpoint", "http://127.0.0.1:2379",
		"comma separated endpoint string for pserver to connect to etcd")
	etcdTimeout := flag.Int("etcd-timeout", 5, "timeout for etcd calls")
	numPservers := flag.Int("num-pservers", 1, "total pserver count in a training job")
	logLevel := flag.String("log-level", "info",
		"log level, possible values: debug, info, warning, error, fatal, panic")
	flag.Parse()

	level, err := log.ParseLevel(*logLevel)
	if err != nil {
		panic(err)
	}
	log.SetLevel(level)

	timeout := time.Second * time.Duration((*etcdTimeout))
	e := pserver.NewEtcdClient(*etcdEndpoint, *numPservers, timeout)
	idx, err := e.Register()
	if err != nil {
		panic(err)
	}

	s, err := pserver.NewService(idx)
	if err != nil {
		panic(err)
	}
	err = rpc.Register(s)
	if err != nil {
		panic(err)
	}

	rpc.HandleHTTP()
	l, err := net.Listen("tcp", ":"+strconv.Itoa(*port))
	if err != nil {
		panic(err)
	}

	log.Infof("start pserver at port %d", *port)
	err = http.Serve(l, nil)

	if err != nil {
		panic(err)
	}
}
