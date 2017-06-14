package master

import (
	"log"
	"time"

	"github.com/PaddlePaddle/Paddle/go/connection"
)

// Addresser provide the address of the master server.
type Addresser interface {
	Address() string
}

// Client is the client of the master server.
type Client struct {
	conn *connection.Conn
}

// NewClient creates a new Client.
func NewClient(addr Addresser) *Client {
	c := &Client{}
	c.conn = connection.New()
	go c.monitorMaster(addr)
	return c
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
					log.Println(err)
				}
			} else {
				err := c.conn.Connect(curMaster)
				if err != nil {
					log.Println(err)

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

// GetTask gets a new task from the master server.
func (c *Client) GetTask() (Task, error) {
	var t Task
	err := c.conn.Call("Service.GetTask", 0, &t)
	return t, err
}

// TaskFinished tells the master server a task is finished.
func (c *Client) TaskFinished(taskID int) error {
	return c.conn.Call("Service.TaskFinished", taskID, nil)
}
