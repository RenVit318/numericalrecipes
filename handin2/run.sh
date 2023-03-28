#!/bin/bash
echo 'Starting Hand-In 2'

if [ ! -d 'results' ]; then
  echo "Directory results does not exist. Creating it.."
  mkdir results
fi

echo 'Running Q1: Satellite Galaxies..'
python3 satellite_galaxies.py

echo 'Running Q2: Heating and Cooling of HII Regions..'
python3 hii_regions.py

echo 'Start Making LaTeX PDF..'

#pdflatex latex/main.tex
#pdflatex latex/main.tex
