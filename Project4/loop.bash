#!/bin/bash
 for threads in 1 2 4 8 16 32
    do
    for size in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
        do
        g++  all04.cpp -DARRAYSIZE=$size -DNUMT=$threads -o all04  -lm  -fopenmp
        ./all04
    done
done
