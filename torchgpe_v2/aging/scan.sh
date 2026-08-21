#!/bin/bash

set -e

temperatures=(15 18 21 24 27 30 32 35 38 41 44 47 50 53  56 59 62 65 68 71 74 77)
particle_numbers=(50000)

for T in "${temperatures[@]}"; do
    for N in "${particle_numbers[@]}"; do

        echo "Submitting T=$T, N=$N"

        ./submit.sh \
            --temperature "$T" \
            --N_particles1 "$N" \
            --N_particles2 20000 \
            --grid_size 60e-6 \
            --omegar 50 \
            --thermalization_time 60e-3 \
            --gamma1 0.01 \
            --gamma2 0.01 \
            --VP 25 \
            --a_s 170

    done
done
