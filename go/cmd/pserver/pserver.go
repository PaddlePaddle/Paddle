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
	index := flag.Int("index", -1, "index of this pserver, should be larger or equal than 0")
	etcdEndpoint := flag.String("etcd-endpoint", "http://127.0.0.1:2379",
		"comma separated endpoint string for pserver to connect to etcd")
	etcdTimeout := flag.Int("etcd-timeout", 5, "timeout for etcd calls")
	numPservers := flag.Int("num-pservers", 1, "total pserver count in a training job")
	checkpointPath := flag.String("checkpoint-path", "/checkpoints/", "save checkpoint path")
	checkpointInterval := flag.Int("checkpoint-interval", 600, "save checkpoint per interval seconds")
	logLevel := flag.String("log-level", "info",
		"log level, possible values: debug, info, warning, error, fatal, panic")
	flag.Parse()

	level, err := log.ParseLevel(*logLevel)
	if err != nil {
		panic(err)
	}
	log.SetLevel(level)

	var idx int
	var cp pserver.Checkpoint
	var e *pserver.EtcdClient
	if *index >= 0 {
		idx = *index
	} else {
		timeout := time.Second * time.Duration((*etcdTimeout))
		e = pserver.NewEtcdClient(*etcdEndpoint, *numPservers, timeout)
		idx, err = e.Register()
		if err != nil {
			panic(err)
		}
	}

	s, err := pserver.NewService(idx, *checkpointInterval, *checkpointPath, e, cp)
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
