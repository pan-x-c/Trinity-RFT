#!/bin/bash

# Usage: ./build_doc.sh [--current-branch]

set -e

if [[ "$1" == "--current-branch" ]]; then
    branch=$(git rev-parse --abbrev-ref HEAD)
    conf_file="source/conf.py"
    cp "${conf_file}" "${conf_file}.bak"
    python3 -c "
import re
path = '${conf_file}'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open(path, 'w', encoding='utf-8') as f:
    for line in lines:
        if line.startswith('smv_branch_whitelist ='):
            f.write(f'smv_branch_whitelist = r\"^(${branch})$\"\\n')
        elif line.startswith('smv_tag_whitelist ='):
            f.write('smv_tag_whitelist = r\"^$\"\\n')
        else:
            f.write(line)
"
    sphinx-multiversion source build/html
    mv "${conf_file}.bak" "${conf_file}"
else
    sphinx-multiversion source build/html
fi
