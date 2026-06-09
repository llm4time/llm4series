#!/bin/bash

# helper script to create symbolic links to
# notebooks in example folder

mkdir -p examples

# remove all
rm -rf examples/*

# cd into website folder
cd examples/ || exit 1

# create symbolic links in website folder
ln -s ../../../examples/* .

# return to initial folder
cd - >/dev/null || exit 1
