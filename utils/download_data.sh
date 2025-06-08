# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

CURRENT_DIR=$(pwd)

mkdir data
cd data

# download datasets and splits
#mit-states
wget -c http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip -O mitstates.zip
#split
wget -c https://www.senthilpurushwalkam.com/publication/compositional/compositional_split_natural.tar.gz -O compositional_split_natural.tar.gz
#cgqa
wget -c https://s3.mlcloud.uni-tuebingen.de/czsl/cgqa-updated.zip -O cgqa.zip


# MIT-States
unzip mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset
rename "s/ /_/g" mit-states/images/*

# C-GQA
unzip cgqa.zip -d cgqa/

# Download new splits for Purushwalkam et. al
tar -zxvf splits.tar.gz

