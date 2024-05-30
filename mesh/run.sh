#!/bin/bash

name="half_rec"

gmsh $name.geo -2 || exit 1
python draw_mesh.py -n $name || exit 1
python fem.py -n $name || exit 1