#!/bin/bash
docker run -d \
	--name keras-full-gpu \
	$(ls /dev/nvidia* | xargs -I{} echo '--device={}') \
	$(ls /usr/lib/*-linux-gnu/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro') \
	-p 8888:8888 \
	-v $(pwd):/srv \
	-v /usr/lib/nvidia-396/:/usr/lib/nvidia-396/ \
	-v /usr/local/cuda-8.0:/usr/local/cuda-8.0 \
	-e LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/lib/nvidia-396/ \
	gw000/keras-full
