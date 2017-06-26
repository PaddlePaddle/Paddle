package pserver

import (
	"time"
	"context"
	"strconv"
	"strings"

	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
)

const DefaultEtcdTimeout time.Duration = time.Second * time.Duration(5)

type pserverEtcdLister struct {
	client    *clientv3.Client
	timeout   time.Duration
	endpoints []string
}

// read ps desired number from etcd.
func(p pserverEtcdLister) desired() int {
	for {
		ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
		resp, err := p.client.Get(ctx, DefaultPsDesiredPath)
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

func(p pserverEtcdLister) List() []Server {
	psDesired := p.desired()

	servers := make([]Server, psDesired)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), p.timeout)
		for i := 0; i < psDesired; i++ {
			psKey := DefaultPsBasePath + strconv.Itoa(i)
			log.Debugf("checking %s", psKey)
			resp, err := p.client.Get(ctx, psKey)
			cancel()
			if err != nil {
				cancel()
				log.Debugf("Get psKey= %s error, %v", psKey, err)
				time.Sleep(p.timeout)
				continue
			}
			kvs := resp.Kvs
			if len(kvs) == 0 {
				log.Infoln("Waiting for ps addr registered ...")
				time.Sleep(p.timeout)
				continue
			}

			psAddr := string(resp.Kvs[0].Value)
			// TODO(Longfei) check the ps address
			if  psAddr == "" {
				cancel()
				log.Debugf("Get psKey = %s,  psAddr is null illegal", psKey, psAddr)
				time.Sleep(p.timeout)
				continue
			}
			log.Debugf("got value (%s) for key: %s", psAddr, psKey)
			servers[i].Index = i
			servers[i].Addr = psAddr
		}
	}
	return servers
}

func NewEtcdAddrLister(endpoints string) (Lister, int) {
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
	}
	log.Debugf("Connected to etcd: %s\n", endpoints)
	lister := pserverEtcdLister{
		client:    cli,
		timeout:   timeout,
		endpoints: ep,
	}
	psDesired := lister.desired()
	return lister, psDesired
}