#!/usr/bin/env bash
source conf.sh


if [[ -z $IMAGE_TAG ]]
then
	echo "No docker tag provided. Cannot run docker image."
else
	echo "Warning: Removing containers with the prefix $CONTAINER_NAME* "
	docker rm -f $CONTAINER_NAME "$CONTAINER_NAME-tensorboard"
	echo "*********************Starting train_or_test.py script.*********************"
	config_file=$1
	docker run --gpus all \
				-d \
				--env-file conf.sh \
				--name $CONTAINER_NAME \
				-v /data/xiao/torch_filter/pyTorch:/tf \
				DEnKF/torch_filter:$IMAGE_TAG \
				/bin/bash -c "python train.py --config ${config_file}"
	echo "*********************Starting tensorboard at localhost:8093*********************"
	logdir="/tf/experiments/$(basename $config_file .yaml)/summaries"
	docker run --name "$CONTAINER_NAME-tensorboard" \
				--gpus=all \
				--network host \
				-d \
				--env-file conf.sh \
				-v /data/xiao/torch_filter/pyTorch:/tf \
				DEnKF/torch_filter:$IMAGE_TAG \
				/bin/bash -c "tensorboard --logdir $logdir --host 0.0.0.0 --port 8093  --reload_multifile=true"
fi