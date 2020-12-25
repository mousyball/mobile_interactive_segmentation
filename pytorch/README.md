# README

* [Introduction](#introduction)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Misc.](#misc)
* [Citation](#citation)

## Introduction

In this work, We extracted the inference part of f-BRS, introduced in [Citation](#citation), and transformed the model into ONNX format. ONNX has a general IR format which makes the migration to production easier.

Futhermore, ONNX Runtime provides the platform-specific optimization. Fortunately, it supports most of platforms by the latest build.

In our case, our application platform is mobile phone. ONNX Runtime do support the intergration with Android NNAPI which makes less effort and seamless intergration.

At last, ONNX transformation is done in this directory only. Android part is in `app/` folder. Thanks for your reading.

## Installation

Install by `Dockerfile`

```bash
docker build -t <image:tag> .
```

Install by `docker-compose.yml` of parent directory.

```bash
docker-compose up -d
```

## Quick Start

Enter the docker

```bash
docker exec -it dc_pytorch /bin/bash
```

If environment is installed by docker, you could activate python environment by pipenv.

```bash
pipenv shell
```

Then, enjoy the works. :)

```bash
python demo_onnx.py --onnx
```

## Misc.

### Python Packages

```bash
# Activate your environment, then
pip install -r requirement.txt
```

### Pretrained Model

**Download from gdrive**

* [f-brs pretrained models](https://github.com/saic-vul/fbrs_interactive_segmentation#pretrained-models)
  * `HRNetV2-W32+OCR` is used in this project.

**Download by script**

```bash
cd ./weights
bash gdown_model.sh
```

## Citation

If you find this work is useful for your research, please cite our paper:

```txt
@article{fbrs2020,
  title={f-BRS: Rethinking Backpropagating Refinement for Interactive Segmentation},
  author={Konstantin Sofiiuk, Ilia Petrov, Olga Barinova, Anton Konushin},
  journal={arXiv preprint arXiv:2001.10331},
  year={2020}
}
```
