#!/bin/bash
module load openmpi

for p in 1 2 4 6 8
do
	mpic++ proj07.cpp -o proj07 -lm
	mpiexec -np $p ./proj07
done
