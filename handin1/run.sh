#!/bin/bash
echo 'Starting Hand-In 1'

if [! -d 'results']; then
  echo "Directory results does not exist. Creating it.."
  mkdir results
fi

# Import data
wget 

# Import any images here

echo 'Running Q1: Poisson Distribution..'
python3 poisson.py

echo 'Running Q2: Vandermonde Matrix'
python3 vandermonde.py

echo 'Start making LaTeX PDF..'

pdflatex latex/main.tex

