package main

import (
	"fmt"
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"strings"
	"time"

	"github.com/namsral/flag"
	log "github.com/sirupsen/logrus"

	"github.com/PaddlePaddle/Paddle/go/master"
	"github.com/PaddlePaddle/Paddle/go/utils/networkhelper"
)

func main() {
	port := flag.Int("port", 8080, "port of the master server.")
	ttlSec := flag.Int("ttl", 60, "etcd lease TTL in seconds.")
	endpoints := flag.String("endpoints", "http://127.0.0.1:2379", "comma separated etcd endpoints. If empty, fault tolerance will not be enabled.")
	taskTimeoutDur := flag.Duration("task_timout_dur", 20*time.Minute, "task timout duration.")
	taskTimeoutMax := flag.Int("task_timeout_max", 3, "max timtout count for each task before it being declared failed task.")
	chunkPerTask := flag.Int("chunk_per_task", 10, "chunk per task.")
	flag.Parse()

	if *endpoints == "" {
		log.Warningln("-endpoints not set, fault tolerance not be enabled.")
	}

	var store master.Store
	if *endpoints != "" {
		eps := strings.Split(*endpoints, ",")
		ip, err := networkhelper.GetExternalIP()
		if err != nil {
			log.Fatal(err)
		}

		addr := fmt.Sprintf("%s:%d", ip, *port)
		store, err = master.NewEtcdClient(eps, addr, master.DefaultLockPath, master.DefaultAddrPath, master.DefaultStatePath, *ttlSec)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		store = &master.InMemStore{}
	}

	s, err := master.NewService(store, *chunkPerTask, *taskTimeoutDur, *taskTimeoutMax)
	if err != nil {
		log.Fatal(err)
	}

	err = rpc.Register(s)
	if err != nil {
		log.Fatal(err)
	}

	rpc.HandleHTTP()
	l, err := net.Listen("tcp", ":"+strconv.Itoa(*port))
	if err != nil {
		log.Fatal(err)
	}

	err = http.Serve(l, nil)
	if err != nil {
		log.Fatal(err)
	}
}
