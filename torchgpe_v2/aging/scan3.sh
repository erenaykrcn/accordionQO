#!/bin/bash

temperatures=(50 75 100 150)
VPs=(4 6 8)
omegas=(40 80 120)

for T in "${temperatures[@]}"; do
    for VP in "${VPs[@]}"; do
        for w in "${omegas[@]}"; do

            echo "Submitting T=$T, VP=$VP, omega_final=$w"

            ./submit3.sh \
                --temperature "$T" \
                --N_particles1 50000 \
                --N_particles2 50000 \
                --grid_size 20e-6 \
                --omega_initial 30 \
                --omega_final "$w" \
                --thermalization_time 50e-3 \
                --gamma 0.001 \
                --VP "$VP" \
                --final_time 50e-3

        done
    done
done

