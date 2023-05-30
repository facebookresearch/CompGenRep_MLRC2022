# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
### Convert tsv data to csv format for transformers training

import re

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("tsv", "", "Input tsv file.")

flags.DEFINE_string("csv", "", "Output csv file.")

def main(unused_argv):
    with open(FLAGS.tsv, 'r') as tsv_file:
        with open(FLAGS.csv, 'w') as csv_file:
            csv_file.write('input\toutput\n')
            for line in tsv_file:
                csv_file.write(line)

if __name__ == "__main__":
  app.run(main)