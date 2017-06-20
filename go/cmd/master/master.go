package main

import (
	"net"
	"net/http"
	"net/rpc"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/namsral/flag"
	log "github.com/sirupsen/logrus"

	"github.com/PaddlePaddle/Paddle/go/master"
)

type inMemStore struct {
	mu  sync.Mutex
	buf []byte
}

func (m *inMemStore) Save(b []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.buf = b
	return nil
}

func (m *inMemStore) Load() ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	return m.buf, nil
}

func main() {
	port := flag.Int("port", 8080, "port of the master server.")

	ttlSec := flag.Int("ttl", 60, "etcd lease TTL in seconds.")
	endpoints := flag.String("endpoints", "", "comma separated etcd endpoints. If empty, fault tolerance will not be enabled.")
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
		var err error
		store, err = master.NewEtcdStore(eps, master.DefaultLockPath, master.DefaultStatePath, *ttlSec)
		if err != nil {
			log.Fatal(err)
		}
	} else {
		store = &inMemStore{}
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
