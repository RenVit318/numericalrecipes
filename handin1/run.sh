#!/bin/bash
echo 'Starting Hand-In 1'

if [! -d 'results']; then
  echo "Directory results does not exist. Creating it.."
  mkdir results
fi

# Import any images here

echo 'Running Q1: Poisson Distribution..'
python poisson.py

read varname
echo 'Running Q2: Vandermonde Matrix'
python vandermonde.py

pdflatex latex/main.tex
read varname