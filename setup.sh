#!/usr/bin/env bash

pip install -r requirements/build.txt

pip install -v -e . #Or  python setup develop



cd ./mmdet/ops/pairwise #compile for the opeartion using in boxinst
python setup.py build develop 

cd ../tree_filter/ #compile for the opeartion using in boxlevelset
python setup.py build develop


