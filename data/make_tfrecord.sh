#!/bin/bash

python3 make_tfrecord_int.py \
	../sc09/train \
	../sc09_tf \
	--name train --labels \
	--ext wav \
	--fs 16000 \
	--nshards 128 \
	--slice_len 1 \

python3 make_tfrecord_int.py \
	../sc09/test \
	../sc09_tf \
	--name test --labels \
	--ext wav \
	--fs 16000 \
	--nshards 128 \
	--slice_len 1 \

python3 make_tfrecord_int.py \
	../sc09/valid \
	../sc09_tf \
	--name valid --labels \
	--ext wav \
	--fs 16000 \
	--nshards 128 \
	--slice_len 1 \
