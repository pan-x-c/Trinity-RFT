#!/bin/bash

# Usage: ./build_doc.sh [--current-branch]

set -e

conf_file="source/conf.py"

cleanup() {
    # Restore the original conf.py if a backup exists
    if [[ -f "${conf_file}.bak" ]]; then
        echo "Restoring original ${conf_file}..."
        mv "${conf_file}.bak" "${conf_file}"
    fi
}

if [[ "$1" == "--current-branch" ]]; then
    # Ensure cleanup runs on script exit or interruption
    trap cleanup EXIT
    branch=$(git rev-parse --abbrev-ref HEAD)

    # Backup the conf file before modifying
    cp "${conf_file}" "${conf_file}.bak"

    # Modify conf.py to build for the current branch only
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
else
    sphinx-multiversion source build/html
fi
