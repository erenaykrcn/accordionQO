#!/bin/bash

set -e

ARGS="$@"

MESSAGE="Automatic commit"

cd ~/Coding/accordionQO

git add -u

if ! git diff --cached --quiet; then
    git commit -m "$MESSAGE"
    git push
else
    echo "Nothing to commit."
fi

ssh euler "bash -l -c '
    set -e

    cd ~/accordionQO
    git pull

    cd torchgpe_v2/aging

    sbatch run_TC.sh $ARGS
'"
