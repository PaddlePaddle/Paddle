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

package main

import (
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"time"

	"github.com/namsral/flag"
	"github.com/topicai/candy"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	log "github.com/sirupsen/logrus"
)

func main() {
	port := flag.Int("port", 0, "port of the pserver")
	index := flag.Int("index", -1, "index of this pserver, should be larger or equal than 0")
	etcdEndpoint := flag.String("etcd-endpoint", "http://127.0.0.1:2379",
		"comma separated endpoint string for pserver to connect to etcd")
	etcdTimeout := flag.Duration("etcd-timeout", 5*time.Second, "timeout for etcd calls")
	numPservers := flag.Int("num-pservers", 1, "total pserver count in a training job")
	checkpointPath := flag.String("checkpoint-path", "/checkpoints/", "save checkpoint path")
	checkpointInterval := flag.Duration("checkpoint-interval", 600*time.Second, "save checkpoint per interval seconds")
	logLevel := flag.String("log-level", "info",
		"log level, possible values: debug, info, warning, error, fatal, panic")
	flag.Parse()

	level, err := log.ParseLevel(*logLevel)
	candy.Must(err)

	log.SetLevel(level)

	var idx int

	var cp pserver.Checkpoint
	var e *pserver.EtcdClient
	if *index >= 0 {
		idx = *index
	} else {
		e = pserver.NewEtcdClient(*etcdEndpoint, *numPservers, *etcdTimeout)
		idx, err = e.Register(*port)
		candy.Must(err)

		cp, err = pserver.NewCheckpointFromFile(*checkpointPath, idx, e)
		if err != nil {
			log.Errorf("Fetch checkpoint failed, %s", err)
		}
	}

	s, err := pserver.NewService(idx, *checkpointInterval, *checkpointPath, e, cp)
	candy.Must(err)

	err = rpc.Register(s)
	candy.Must(err)

	rpc.HandleHTTP()
	l, err := net.Listen("tcp", ":"+strconv.Itoa(*port))
	candy.Must(err)

	log.Infof("start pserver at port %d", *port)
	err = http.Serve(l, nil)
	candy.Must(err)
}
