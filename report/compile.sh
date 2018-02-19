#!/usr/bin/bash

pdflatex mlp-cw1-template.tex
bibtex mlp-cw1-template.aux
pdflatex mlp-cw1-template.tex
pdflatex mlp-cw1-template.tex