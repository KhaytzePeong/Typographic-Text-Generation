# Typographic Text Generation

Implementation of [Typographic Text Generation with Off-the-Shelf Diffusion Model](https://arxiv.org/abs/2402.14314).

A combination of the off-the-shelf methods ([ControlNet](https://arxiv.org/abs/2302.05543) & [Blended Latent Diffusion](https://arxiv.org/abs/2206.02779)) for typographic text generation.

Refer `demo.ipynb` for examples of some use cases.

# Environment and Pretrained Models (from [ControlNet's official repository](https://github.com/lllyasviel/ControlNet/tree/main/github_page))

First create a new conda environment

    conda env create -f environment.yaml
    conda activate control

All models and detectors can be downloaded from [our Hugging Face page](https://huggingface.co/lllyasviel/ControlNet). Make sure that SD models are put in "ControlNet/models" and detectors are put in "ControlNet/annotator/ckpts". 

In this work, we used the canny model for text generation.
    ControlNet/models/control_sd15_canny.pth

For more information on ControlNet, refer to the [official repository](https://github.com/lllyasviel/ControlNet/tree/main/github_page)


[Arxiv Link](https://arxiv.org/abs/2402.14314)

