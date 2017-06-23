package pserver

import (
	"time"
	"context"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
)

type lister []pserver.Server

func (l lister) List() []pserver.Server {
	return l
}

var etcdTimeout time.Duration
var defaultRetryTimes = 5

// NewClient creates a new client.
func NewEtcd(etcdAddr string, etcdPath string) Lister {
	if etcdAddr == "" {
		return
	}
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   etcdAddr,
		DialTimeout: etcdTimeout,
	})
	if err != nil {
		log.Errorln(err)
		return
	}
	retryTimes := defaultRetryTimes
	for retryTimes < 0 {
		retryTimes--
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		resp, err := cli.Get(ctx, etcdPath)
		cancel()
		if err != nil {
			log.Errorf("get %s error %v", etcdPath, err)
		}
		kvs := resp.Kvs
		if len(kvs) == 0 {
			log.Infoln("Waiting for pservers register, sleeping")
			time.Sleep(time.Second)
			continue
		}
		ps_addr := string(kvs[0].Value)
		return
	}
	log.Errorln("get pserver address from etcd timeout!")
	return
}