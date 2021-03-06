#!/bin/bash

sizes=(1 3 4 7 9 15 16 17 31 33 41)
batch_size=12

for n in "${sizes[@]}" 
do
    for m in "${sizes[@]}" 
    do
        for k in "${sizes[@]}"
        do
            echo $n $m $k 1>&2
            (./TestSimulation $n $m $k | tee sim_output.${n}.${m}.${k}.txt) &

            if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
                wait -n
            fi
        done
    done
done

wait