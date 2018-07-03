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
	"fmt"
	"net"
	"net/http"
	"net/rpc"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"

	log "github.com/inconshreveable/log15"
	"github.com/namsral/flag"

	"github.com/PaddlePaddle/Paddle/go/master"
	"github.com/PaddlePaddle/Paddle/go/utils/networkhelper"
)

func main() {
	port := flag.Int("port", 8080, "port of the master server.")
	ttlSec := flag.Int("ttl", 60, "etcd lease TTL in seconds.")
	endpoints := flag.String("endpoints", "http://127.0.0.1:2379", "comma separated etcd endpoints. If empty, fault tolerance will not be enabled.")
	taskTimeoutDur := flag.Duration("task-timout-dur", 20*time.Minute, "task timout duration.")
	taskTimeoutMax := flag.Int("task-timeout-max", 3, "max timtout count for each task before it being declared failed task.")
	chunkPerTask := flag.Int("chunk-per-task", 10, "chunk per task.")
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

	if *endpoints == "" {
		log.Warn("-endpoints not set, fault tolerance not be enabled.")
	}

	var store master.Store
	if *endpoints != "" {
		eps := strings.Split(*endpoints, ",")
		ip, err := networkhelper.GetExternalIP()
		if err != nil {
			log.Crit("get external ip error", log.Ctx{"error": err})
			panic(err)
		}

		addr := fmt.Sprintf("%s:%d", ip, *port)
		store, err = master.NewEtcdClient(eps, addr, master.DefaultLockPath, master.DefaultAddrPath, master.DefaultStatePath, *ttlSec)
		if err != nil {
			log.Crit("error creating etcd client.", log.Ctx{"error": err})
			panic(err)
		}
	} else {
		store = &master.InMemStore{}
	}

	shutdown := func() {
		log.Info("shutting down gracefully")
		err := store.Shutdown()
		if err != nil {
			log.Error("shutdown error", log.Ctx{"error": err})
		}
	}

	// Guaranteed to run even panic happens.
	defer shutdown()

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)

	s, err := master.NewService(store, *chunkPerTask, *taskTimeoutDur, *taskTimeoutMax)
	if err != nil {
		log.Crit("error creating new service.", log.Ctx{"error": err})
		panic(err)
	}

	err = rpc.Register(s)
	if err != nil {
		log.Crit("error registering to etcd.", log.Ctx{"error": err})
		panic(err)
	}

	rpc.HandleHTTP()
	l, err := net.Listen("tcp", ":"+strconv.Itoa(*port))
	if err != nil {
		log.Crit("error listing to port", log.Ctx{"error": err, "port": *port})
		panic(err)
	}

	go func() {
		err = http.Serve(l, nil)
		if err != nil {
			log.Crit("error serving HTTP", log.Ctx{"error": err})
			panic(err)
		}
	}()

	<-c
}
