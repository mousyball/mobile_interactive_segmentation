# README

* [Installation](#installation)

## TODO

* pytorch model
  * [x] inference
  * [x] ONNX
* mobile
  * hello world
    * [ ] basic UI
    * [ ] lenet + cifar10 inference
  * [ ] cpu pre-processing and post-processing
  * [ ] NNAPI integration
  * [ ] test with mocking data
  * [ ] integration with interactive UI
* environment
  * [x] docker
  * [x] pipenv
  * [ ] mobile env/package
* docs


## Installation

Firstly, setup the environment by docker-compose.

```bash
# Build in detach mode
docker-compose up -d
```

For instance, There are two containers at the moment.

* `dc_pytorch`
* `dc_android`

Then, enter the container you want to use.

```bash
docker exec -it dc_pytorch /bin/bash
```
