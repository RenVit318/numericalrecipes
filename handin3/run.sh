#!/bin/bash
echo 'Starting Hand-In 3'

if [ ! -d 'results' ]; then
  echo "Directory results does not exist. Creating it.."
  mkdir results
fi

echo 'Downloading Data'
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m11.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m12.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m13.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m14.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m15.txt

echo 'Running Code: Satellite Galaxies..'
python3 galaxies.py

echo 'Start Making LaTeX PDF..'

pdflatex latex/main.tex
pdflatex latex/main.tex
