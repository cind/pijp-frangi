#!/bin/bash


cases=$(cat cleaned_reference_cases.txt)
for c in $cases
do
    echo "Running case $c"
    python frangi.py -p ADNI3_frangi -s stage -c $c
done
