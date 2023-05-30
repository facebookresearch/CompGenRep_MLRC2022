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
"""Measures and prints compound divergence between two sets of examples."""

from absl import app
from absl import flags

import sys
import os
sys.path.append(os.getenv("BASE_DIR")+"/baseline_replication/TMCD")
from tasks import mcd_utils
from tasks import tsv_utils
from tasks.geoquery import tmcd_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("input_1", "", "Input tsv file.")

flags.DEFINE_string("input_2", "", "Input tsv file.")


def main(unused_argv):
  examples_1 = tsv_utils.read_tsv(FLAGS.input_1)
  examples_2 = tsv_utils.read_tsv(FLAGS.input_2)
  divergence = mcd_utils.measure_example_divergence(
      examples_1, examples_2, get_compounds_fn=tmcd_utils.get_example_compounds)
  print("Compound divergence: %s" % divergence)


if __name__ == "__main__":
  app.run(main)
