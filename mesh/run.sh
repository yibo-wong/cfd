#!/bin/bash

gmsh cylinder.geo -2 || exit 1
python draw_mesh.py || exit 1