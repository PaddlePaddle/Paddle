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

package client

import (
	"context"
	"strconv"
	"strings"
	"time"

	"github.com/PaddlePaddle/Paddle/go/pserver"
	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
)

const (
	defaultEtcdTimeout time.Duration = 5 * time.Second
)

// EtcdClient is used by pserver client that is a part of trainer process.
// TODO:
// 1. add watcher to watch the change state of pservers)
// 1. add etcd lock)
type EtcdClient struct {
	client    *clientv3.Client
	timeout   time.Duration
	endpoints []string
}

// Desired read ps desired number from etcd.
func (p *EtcdClient) Desired() int {
	var psDesired int
	for {
		ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
		resp, err := p.client.Get(ctx, pserver.PsDesired)
		cancel()
		if err != nil {
			log.Errorf("Get ps dresire number failed! recnnectiong..., %v", err)
			time.Sleep(p.timeout)
			continue
		}

		kvs := resp.Kvs
		if len(kvs) == 0 {
			log.Infoln("Waiting for ps desired registered ...")
			time.Sleep(p.timeout)
			continue
		}

		psDesired, err = strconv.Atoi(string(resp.Kvs[0].Value))
		if err != nil {
			log.Errorf("psDesired %d invalid %v", psDesired, err)
			time.Sleep(p.timeout)
			continue
		}

		log.Debugf("Get psDesired number: %d", psDesired)
		break
	}
	return psDesired
}

// List return the pserver list read from etcd.
func (p *EtcdClient) List() []Server {
	psDesired := p.Desired()

	servers := make([]Server, psDesired)
	for {
		for i := 0; i < psDesired; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
			psKey := pserver.PsPath + strconv.Itoa(i)
			log.Debugf("checking %s", psKey)
			resp, err := p.client.Get(ctx, psKey)
			cancel()
			if err != nil {
				log.Infof("Get psKey= %s error, %v", psKey, err)
				time.Sleep(p.timeout)
				continue
			}
			kvs := resp.Kvs
			if len(kvs) == 0 {
				log.Infof("Waiting for ps addr registered ...")
				time.Sleep(p.timeout)
				continue
			}

			psAddr := string(resp.Kvs[0].Value)
			// TODO(Longfei) check the ps address
			if psAddr == "" {
				log.Infof("Get psKey = %s, psAddr is empty", psKey)
				time.Sleep(p.timeout)
				continue
			}
			log.Infof("got value (%s) for key: %s", psAddr, psKey)
			servers[i].Index = i
			servers[i].Addr = psAddr
		}
		break
	}
	return servers
}

// NewEtcd create a etcd client to return the state of pserver on etcd.
func NewEtcd(endpoints string) *EtcdClient {
	ep := strings.Split(endpoints, ",")
	var cli *clientv3.Client
	var err error
	for {
		cli, err = clientv3.New(clientv3.Config{
			Endpoints:   ep,
			DialTimeout: defaultEtcdTimeout,
		})
		if err != nil {
			log.Errorf("Init etcd connection failed: %v", err)
			time.Sleep(defaultEtcdTimeout)
			continue
		}
		break
	}
	log.Infof("Connected to etcd: %s\n", endpoints)
	client := &EtcdClient{
		client:    cli,
		timeout:   defaultEtcdTimeout,
		endpoints: ep,
	}
	return client
}
