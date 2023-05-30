# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
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
"""Function for loading config json file."""

import json

from tensorflow.io import gfile


def json_file_to_dict(json_file):
  """Constructs a dictionary from a json file."""
  with gfile.GFile(json_file, "r") as reader:
    text = reader.read()
  return json.loads(text)
