#!/bin/bash

python make_tfrecord_int.py \
	new/sc09/train \
	new/sc09_tf \
	--name train --labels \
	--ext wav \
	--fs 16000 \
	--nshards 128 \
	--slice_len 1 \

python make_tfrecord_int.py \
	new/sc09/test \
	new/sc09_tf \
	--name test --labels \
	--ext wav \
	--fs 16000 \
	--nshards 128 \
	--slice_len 1 \

python make_tfrecord_int.py \
	new/sc09/valid \
	new/sc09_tf \
	--name valid --labels \
	--ext wav \
	--fs 16000 \
	--nshards 128 \
	--slice_len 1 \
