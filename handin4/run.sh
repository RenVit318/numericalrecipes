#!/bin/bash
echo 'Starting Hand-In 3'

if [ ! -d 'results' ]; then
  echo "Directory results does not exist. Creating it.."
  mkdir results
fi


echo 'Running A1: Solar System Orbits..'
python3 orbits.py

echo 'Running A2: Fourier Forces..'
python3 fourier.py

echo 'Running A3: Galaxy Classification..'
wget https://home.strw.leidenuniv.nl/~belcheva/galaxy_data.txt
python3 classification.py

echo 'Start Making LaTeX PDF..'

pdflatex latex/main.tex
pdflatex latex/main.tex
