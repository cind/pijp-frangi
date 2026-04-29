#!/bin/bash

while IFS= read -r subject; do
    if [ -z "$subject" ]; then
        continue
    fi
    echo "Processing $subject..."
    python pijp-mcpvs_preprocess.py -p ADNI3_frangi -s preprocess -c "$subject" --nogrid
    echo "Done with $subject"
done < redo_subjects.txt