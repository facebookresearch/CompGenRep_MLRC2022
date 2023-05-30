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
"""Convert SCAN txt format to standard TSV format."""

from absl import app
from absl import flags

import sys
import os
sys.path.append(os.getenv("BASE_DIR")+"/baseline_replication/TMCD")
from tasks import tsv_utils

from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", "Input txt file.")

flags.DEFINE_string("output", "", "Output tsv file.")


def load_examples(filename):
  """Load SCAN examples from original data file."""
  examples = []

  with gfile.GFile(filename, "r") as input_file:
    for line in input_file:
      splits = line.split("OUT:")
      # Trim "IN:" prefix.
      input_string = splits[0][3:].strip()
      output_string = splits[1].strip()
      examples.append((input_string, output_string))

  return examples


def main(unused_argv):
  examples = load_examples(FLAGS.input)
  tsv_utils.write_tsv(examples, FLAGS.output)


if __name__ == "__main__":
  app.run(main)
