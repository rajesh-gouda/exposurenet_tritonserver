## Triton server for exposurenet

# How to use
1. Clone this repository
2. build the docker image using `docker build -t tritonserver-torch -f Dockerfile.torch .`
3. run the docker using `docker run -it --name triton_server --net=host   -v $(pwd)/model_repository:/models   tritonserver-torch   tritonserver --model-repository=/models`
this will start the triton server