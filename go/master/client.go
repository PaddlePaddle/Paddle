package master

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/PaddlePaddle/Paddle/go/connection"
	"github.com/PaddlePaddle/recordio"
	"github.com/coreos/etcd/clientv3"
	log "github.com/sirupsen/logrus"
)

const masterAddrPath = "/master"

// Addresser provide the address of the master server.
type Addresser interface {
	Address() string
}

// Client is the client of the master server.
type Client struct {
	conn *connection.Conn
	ch   chan []byte
}

// MasterAddresser provide master address
type masterAddresser struct {
	client    *clientv3.Client
	timeout   time.Duration
	endpoints []string
}

// Address return the address
func (m masterAddresser) Address() string {
	for {
		ctx, cancel := context.WithTimeout(context.Background(), m.timeout)
		resp, err := m.client.Get(ctx, masterAddrPath)
		cancel()
		if err != nil {
			log.Errorf("Fetch master addr failed, reconnecting to etcd, %v", err)
			err := m.client.Close()
			if err != nil {
				log.Errorln(err)
				time.Sleep(m.timeout)
				continue
			}
			// reconnect to etcd server
			m.client, err = clientv3.New(clientv3.Config{
				Endpoints:   m.endpoints,
				DialTimeout: m.timeout,
			})
			if err != nil {
				log.Errorf("Reconnecting etcd failed, sleep for %d seconds ...\n%v", m.timeout, err)
				time.Sleep(m.timeout)
				continue
			}
			continue
		}
		kvs := resp.Kvs
		if len(kvs) == 0 {
			log.Infoln("Waiting for master be ready ...")
			time.Sleep(m.timeout)
			continue
		}
		mAddr := kvs[0].Value
		log.Debugf("Fetched master address: %s\n", mAddr)
		return string(mAddr)
	}
}

// NewClient creates a new Client.
//
// bufSize is the record buffer size. NextRecord will read from this
// buffer.
func NewClient(addr Addresser, bufSize int) *Client {
	c := &Client{}
	c.conn = connection.New()
	c.ch = make(chan []byte, bufSize)
	go c.monitorMaster(addr)
	go c.getRecords()
	return c
}

// NewEtcdClient create a new master client by etcd
//
// endpoints is the endpoints for etcd and separated by ",", such as
// "172.0.1.0:2379,172.0.1.1:2379"
// timeout is the timeout for etcd calls
// bufSize is the record buffer size. NextRecord will read from this buffer.
func NewEtcdClient(endpoints string, timeout int, bufSize int) *Client {
	t := time.Second * time.Duration(timeout)
	ep := strings.Split(endpoints, ",")
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   ep,
		DialTimeout: t,
	})
	if err != nil {
		log.Errorf("Init etcd connection failed: %v", err)
		panic(err)
	}
	log.Debugf("Connected to etcd: %s\n", endpoints)
	mAddresser := masterAddresser{
		client:    cli,
		timeout:   t,
		endpoints: ep,
	}
	return NewClient(mAddresser, bufSize)
}

func (c *Client) getRecords() {
	for {
		t, err := c.getTask()
		if err != nil {
			// TODO(helin): wait before move on with next
			// getTask call.
			log.Errorln(err)
			continue
		}

		for _, chunk := range t.Chunks {
			f, err := os.Open(chunk.Path)
			if err != nil {
				log.Errorln(err)
				continue
			}

			s := recordio.NewRangeScanner(f, &chunk.Index, -1, -1)
			for s.Scan() {
				c.ch <- s.Record()
			}

			if s.Err() != nil {
				log.Errorln(err, chunk.Path)
			}

			err = f.Close()
			if err != nil {
				log.Errorln(err)
			}
		}

		// We treat a task as finished whenever the last data
		// instance of the task is read. This is not exactly
		// correct, but a reasonable approximation.
		c.taskFinished(t.ID)
	}
}

func (c *Client) monitorMaster(addr Addresser) {
	lastMaster := ""
	monitor := func() {
		// get the lastest address of the master server,
		// connect to the new address once address changed.
		curMaster := addr.Address()
		if curMaster != lastMaster {
			if curMaster == "" {
				err := c.conn.Close()
				if err != nil {
					log.Errorln(err)
				}
			} else {
				err := c.conn.Connect(curMaster)
				if err != nil {
					log.Errorln(err)

					// connect to addr failed, set
					// to last known addr in order
					// to retry next time.
					curMaster = lastMaster
				}

			}
		}

		lastMaster = curMaster
	}

	monitor()
	ticker := time.NewTicker(10 * time.Second)
	for _ = range ticker.C {
		monitor()
	}
}

// SetDataset set dataset for the master server to dispatch.
//
// SetDataset can be call multiple times from different nodes. But
// only the first call will be honored.
func (c *Client) SetDataset(globPaths []string) error {
	return c.conn.Call("Service.SetDataset", globPaths, nil)
}

// getTask gets a new task from the master server.
func (c *Client) getTask() (Task, error) {
	var t Task
	err := c.conn.Call("Service.GetTask", 0, &t)
	return t, err
}

// TaskFinished tells the master server a task is finished.
func (c *Client) taskFinished(taskID int) error {
	return c.conn.Call("Service.TaskFinished", taskID, nil)
}

// NextRecord returns next record in the dataset.
//
// NextRecord will block until the next record is available. It is
// thread-safe.
func (c *Client) NextRecord() []byte {
	return <-c.ch
}
