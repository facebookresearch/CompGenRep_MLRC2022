# Copyright (c) Meta Platforms, Inc. and affiliates All Rights Reserved
### Convert tsv data to csv format for transformers training

import re

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("tsv", "", "Input tsv file.")

flags.DEFINE_string("output", "", "Output tsv file.")

def main(unused_argv):
    with open(FLAGS.tsv, 'r') as tsv_file:
        with open(FLAGS.output, 'w') as csv_file:
            for line in tsv_file:
                # remove the type column
                csv_file.write(line.split("\t")[0] + "\t" + line.split("\t")[1] + "\n")
    print("Writting done")

if __name__ == "__main__":
  app.run(main)