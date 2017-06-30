package pserver

import (
	"context"
	"strconv"
	"strings"
	"time"

	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
)

const (
	DefaultEtcdTimeout time.Duration = 5 * time.Second
)

type EtcdCClient interface {
	Desired() int
	List() []Server
}

// TODO(Longfei)
// 1. add watcher to watch the change state of pservers)
// 1. add etcd lock)
type EtcdCClientImpl struct {
	client    *clientv3.Client
	timeout   time.Duration
	endpoints []string
}

// read ps desired number from etcd.
func (p *EtcdCClientImpl) Desired() int {
	for {
		ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
		resp, err := p.client.Get(ctx, PsDesired)
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

		psDesired, err := strconv.Atoi(string(resp.Kvs[0].Value))
		if err != nil {
			log.Errorf("psDesired %s invalid %v", psDesired, err)
			time.Sleep(p.timeout)
			continue
		}

		log.Debugf("Get psDesired number: %d\n", psDesired)
		return psDesired
	}
}

func (p *EtcdCClientImpl) List() []Server {
	psDesired := p.Desired()

	servers := make([]Server, psDesired)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
		for i := 0; i < psDesired; i++ {
			psKey := PsPath + strconv.Itoa(i)
			log.Debugf("checking %s", psKey)
			resp, err := p.client.Get(ctx, psKey)
			if err != nil {
				cancel()
				log.Infof("Get psKey= %s error, %v", psKey, err)
				time.Sleep(p.timeout)
				continue
			}
			kvs := resp.Kvs
			if len(kvs) == 0 {
				cancel()
				log.Infof("Waiting for ps addr registered ...")
				time.Sleep(p.timeout)
				continue
			}

			psAddr := string(resp.Kvs[0].Value)
			// TODO(Longfei) check the ps address
			if psAddr == "" {
				cancel()
				log.Infof("Get psKey = %s, psAddr is empty", psKey)
				time.Sleep(p.timeout)
				continue
			}
			log.Infof("got value (%s) for key: %s", psAddr, psKey)
			servers[i].Index = i
			servers[i].Addr = psAddr
		}
		cancel()
		break
	}
	return servers
}

func NewEtcdCClient(endpoints string) (EtcdCClient, error) {
	ep := strings.Split(endpoints, ",")
	timeout := DefaultEtcdTimeout
	var cli *clientv3.Client
	var err error
	for {
		cli, err = clientv3.New(clientv3.Config{
			Endpoints:   ep,
			DialTimeout: timeout,
		})
		if err != nil {
			log.Errorf("Init etcd connection failed: %v", err)
			time.Sleep(timeout)
			continue
		}
		break
	}
	log.Infof("Connected to etcd: %s\n", endpoints)
	client := &EtcdCClientImpl{
		client:    cli,
		timeout:   timeout,
		endpoints: ep,
	}
	return client, nil
}
