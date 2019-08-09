/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

package com.baidu.paddle.lite;

/**
 * CxxConfig is the configuration for the Full feature predictor.
 */
public class CxxConfig extends ConfigBase {

    Place preferredPlace;
    Place[] validPlaces;

    public Place getPreferredPlace() {
        return preferredPlace;
    }

    public void setPreferredPlace(Place preferredPlace) {
        this.preferredPlace = preferredPlace;
    }

    public Place[] getValidPlaces() {
        return validPlaces;
    }

    public void setValidPlaces(Place[] validPlaces) {
        this.validPlaces = validPlaces;
    }
}
