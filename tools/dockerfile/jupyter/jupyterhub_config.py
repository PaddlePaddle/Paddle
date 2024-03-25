# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for jupyterhub.

#------------------------------------------------------------------------------
# JupyterHub(Application) configuration
#------------------------------------------------------------------------------

## An Application for starting a Multi-User Jupyter Notebook server.

import os
env_dict = os.environ

if "USER_PASSWD" in env_dict:
    passwd = env_dict["USER_PASSWD"]
else:
    passwd = "hipaddle"

## Class for authenticating users.
#  
#  This should be a class with the following form:
#  
#  - constructor takes one kwarg: `config`, the IPython config object.
#  
#  with an authenticate method that:
#  
#  - is a coroutine (asyncio or tornado)
#  - returns username on success, None on failure
#  - takes two arguments: (handler, data),
#    where `handler` is the calling web.RequestHandler,
#    and `data` is the POST form data from the login page.
c.JupyterHub.authenticator_class = 'dummyauthenticator.DummyAuthenticator'
c.DummyAuthenticator.password = passwd
#c.DummyAuthenticator.add_user_cmd = ['adduser', '-q', '--gecos', '""', '--home', '/home/USERNAME', '--disabled-password']
#c.DummyAuthenticator.create_system_users = True

## The public facing URL of the whole JupyterHub application.
#  
#  This is the address on which the proxy will bind. Sets protocol, ip, base_url
c.JupyterHub.bind_url = 'http://0.0.0.0:80/'

## File in which to store the cookie secret.
c.JupyterHub.cookie_secret_file = '/srv/jupyterhub_cookie_secret'

#------------------------------------------------------------------------------
# Authenticator(LoggingConfigurable) configuration
#------------------------------------------------------------------------------

## Base class for implementing an authentication provider for JupyterHub

## Set of users that will have admin rights on this JupyterHub.
#  
#  Admin users have extra privileges:
#   - Use the admin panel to see list of users logged in
#   - Add / remove users in some authenticators
#   - Restart / halt the hub
#   - Start / stop users' single-user servers
#   - Can access each individual users' single-user server (if configured)
#  
#  Admin access should be treated the same way root access is.
#  
#  Defaults to an empty set, in which case no user has admin access.
c.Authenticator.admin_users = set(("jovyan", "paddle"))

## Whitelist of usernames that are allowed to log in.
#  
#  Use this with supported authenticators to restrict which users can log in.
#  This is an additional whitelist that further restricts users, beyond whatever
#  restrictions the authenticator has in place.
#  
#  If empty, does not perform any additional restriction.
c.Authenticator.whitelist = set(("jovyan", "paddle"))

#------------------------------------------------------------------------------
# LocalAuthenticator(Authenticator) configuration
#------------------------------------------------------------------------------

## Base class for Authenticators that work with local Linux/UNIX users
#  
#  Checks for local users, and can attempt to create them if they exist.

## The command to use for creating users as a list of strings
#  
#  For each element in the list, the string USERNAME will be replaced with the
#  user's username. The username will also be appended as the final argument.
#  
#  For Linux, the default value is:
#  
#      ['adduser', '-q', '--gecos', '""', '--disabled-password']
#  
#  To specify a custom home directory, set this to:
#  
#      ['adduser', '-q', '--gecos', '""', '--home', '/customhome/USERNAME', '--
#  disabled-password']
#  
#  This will run the command:
#  
#      adduser -q --gecos "" --home /customhome/river --disabled-password river
#  
#  when the user 'river' is created.
c.LocalAuthenticator.add_user_cmd = [
    'adduser', '-q', '--gecos', '""', '--home', '/home/USERNAME',
    '--disabled-password'
]

## If set to True, will attempt to create local system users if they do not exist
#  already.
#  
#  Supports Linux and BSD variants only.
c.LocalAuthenticator.create_system_users = True
