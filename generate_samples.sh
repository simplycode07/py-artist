#!/bin/bash

input_file="input.jpg"

for block_size in 8 16; do
	for effective_area in 3 6 ; do
		for strength_multiplier in 0.5 1 2; do
			echo $input_file $block_size $effective_area $strength_multiplier
			python main.py $input_file $block_size $effective_area $strength_multiplier
		done
	done
done
