#!/bin/bash

wget http://miniplaces.csail.mit.edu/data/data.tar.gz
tar -xvf data.tar.gz

mv images/ data/
mv objects/ data/
rm data.tar.gz