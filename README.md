# Triton server for exposurenet

## How to use
1. Clone this repository
2. build the docker image using `docker build -t tritonserver-torch -f Dockerfile.torch .`
3. run the docker using `docker run -it --name triton_server --net=host   -v $(pwd)/model_repository:/models   tritonserver-torch   tritonserver --model-repository=/models`
this will start the triton server

### POST Request from postman
1. url for api `http://localhost:8000/v2/models/exposurenet/infer` update the ip
2. create a json body with details like `{
  "inputs": [
    {
      "name": "input_str",
      "shape": [1],
      "datatype": "BYTES",
      "data": [
        "{\"delivery_days\": 12, \"num_words\": 300, \"num_sentences\": 20, \"num_emotional_shifts\": 3, \"num_conflict_scenes\": 2, \"num_plot_twists\": 1, \"num_drama_hooks\": 1, \"length\": 120, \"associated_copy_original_language\": \"english\", \"genre\": \"drama\", \"delivery_media\": \"tv,youtube\", \"delivery_country\": \"india,usa\"}"
      ]
    }
  ]
}
`