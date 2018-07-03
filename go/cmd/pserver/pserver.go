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

package main

import (
	"net"
	"net/http"
	"net/rpc"
	"os"
	"os/signal"
	"strconv"
	"time"

	"github.com/namsral/flag"
	"github.com/topicai/candy"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	log "github.com/inconshreveable/log15"
)

func main() {
	port := flag.Int("port", 8001, "port of the pserver")
	index := flag.Int("index", -1, "index of the pserver, set to -1 if use etcd for auto pserver index registry")
	etcdEndpoint := flag.String("etcd-endpoint", "http://127.0.0.1:2379",
		"comma separated endpoint string for pserver to connect to etcd")
	dialTimeout := flag.Duration("dial-timeout", 5*time.Second, "dial timeout")
	etcdTTL := flag.Int("etcd-ttl", 5, "etcd time to live in seconds")
	numPservers := flag.Int("num-pservers", 1, "total pserver count in a training job")
	checkpointPath := flag.String("checkpoint-path", "/checkpoints/", "save checkpoint path")
	checkpointInterval := flag.Duration("checkpoint-interval", 600*time.Second, "save checkpoint per interval seconds")
	logLevel := flag.String("log-level", "info",
		"log level, possible values: debug, info, warn, error, crit")
	flag.Parse()

	lvl, err := log.LvlFromString(*logLevel)
	if err != nil {
		panic(err)
	}

	log.Root().SetHandler(
		log.LvlFilterHandler(lvl, log.CallerStackHandler("%+v", log.StderrHandler)),
	)

	var idx int

	var cp pserver.Checkpoint
	var e *pserver.EtcdClient
	if *index >= 0 {
		idx = *index
	} else {
		e = pserver.NewEtcdClient(*etcdEndpoint, *numPservers, *dialTimeout, *etcdTTL)
		idx, err = e.Register(*port)
		candy.Must(err)

		cp, err = pserver.LoadCheckpoint(e, idx)
		if err != nil {
			if err == pserver.ErrCheckpointNotFound {
				log.Info("load checkpoint error", "error", err)
			} else {
				panic(err)
			}
		}
	}

	shutdown := func() {
		log.Info("shutting down gracefully")
		sErr := e.Shutdown()
		if sErr != nil {
			log.Error("error shutting down", log.Ctx{"error": sErr})
		}
	}

	// Guaranteed to run even panic happens.
	defer shutdown()

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)

	s, err := pserver.NewService(idx, *checkpointInterval, *checkpointPath, e, cp)
	candy.Must(err)

	err = rpc.Register(s)
	candy.Must(err)

	rpc.HandleHTTP()
	l, err := net.Listen("tcp", ":"+strconv.Itoa(*port))
	candy.Must(err)

	go func() {
		log.Info("serving pserver", log.Ctx{"port": *port})
		err = http.Serve(l, nil)
		candy.Must(err)
	}()

	<-c
}
