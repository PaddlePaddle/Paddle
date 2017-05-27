package pserver

// Client is the client to parameter servers.
type Client struct {
}

// NewClient creates a new client.
func NewClient(addr string) *Client {
	return &Client{}
}

// BeginInitParams begins to initialize parameters on parameter
// servers.
//
// BeginInitParams will be called from multiple trainers, only one
// trainer will be selected to initialize the parameters on parameter
// servers. Other trainers will be blocked until the initialization is
// done, and they need to get the initialized parameters from
// parameter servers using GetParams.
func (c *Client) BeginInitParams(pserverConfigProto []byte) (selected bool, err error) {
	return true, nil
}

// InitParam initializes the parameter on parameter servers.
func (c *Client) InitParam(paramWithConfigs ParameterWithConfig) error {
	return nil
}

// FinishInitParams tells parameter servers client has sent all
// parameters to parameter servers as initialization.
func (c *Client) FinishInitParams() error {
	return nil
}

// SendGrads sends gradients to parameter servers for updating
// parameters.
func (c *Client) SendGrads(grads []Gradient) error {
	return nil
}

// GetParams gets parameters from parameter servers.
func (c *Client) GetParams(names []string) ([]Parameter, error) {
	return nil, nil
}

// SaveModel indicates parameters to save the parameter to the given
// path.
func (c *Client) SaveModel(path string) error {
	return nil
}

// Cleanup cleans up the client states.
func (c *Client) Cleanup() {
}
