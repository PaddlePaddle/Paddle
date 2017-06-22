package networkhelper

import "testing"

func TestGetIP(t *testing.T) {
	_, err := GetExternalIP()
	if err != nil {
		t.Errorf("GetExternalIP returns error : %v\n", err)
	}
}
