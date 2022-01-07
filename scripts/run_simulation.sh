#!/bin/bash

sizes=(1 3 4 7 9 15 16 17 31 33 41)
small_sizes=(1 3 4 7 9 15 16 17)
large_sizes=(31 33 41)
batch_size=12


for n in "${small_sizes[@]}" 
do
    for m in "${small_sizes[@]}" 
    do
        for k in "${small_sizes[@]}"
        do
            echo $n $m $k 1>&2
            (./TestSimulation $n $m $k | tee sim_output.${n}.${m}.${k}.txt) &

            if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
                wait -n
            fi
        done
    done
done


for n in "${small_sizes[@]}" 
do
    for m in "${large_sizes[@]}" 
    do
        for k in "${large_sizes[@]}"
        do
            echo $n $m $k 1>&2
            (./TestSimulation $n $m $k | tee sim_output.${n}.${m}.${k}.txt) &

            if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
                wait -n
            fi
        done
    done
done

for n in "${large_sizes[@]}" 
do
    for m in "${small_sizes[@]}" 
    do
        for k in "${large_sizes[@]}"
        do
            echo $n $m $k 1>&2
            (./TestSimulation $n $m $k | tee sim_output.${n}.${m}.${k}.txt) &

            if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
                wait -n
            fi
        done
    done
done

for n in "${large_sizes[@]}" 
do
    for m in "${large_sizes[@]}" 
    do
        for k in "${small_sizes[@]}"
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