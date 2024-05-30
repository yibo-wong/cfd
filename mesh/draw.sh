#!/bin/bash

gmsh full_rec.geo -2 || exit 1
python draw_mesh.py -n full_rec || exit 1