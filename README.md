______________________________________________________________________

<div align="center">

# Diffusion Models

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

My implementation of Latent Diffusion Models. The codebase includes two samplers: DDPM and DDIM.

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/phamgialinhlx/diffusion-model
cd diffusion-model

# [OPTIONAL] create conda environment
conda create -n diffusion python=3.8 -y
conda activate diffusion

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train_ae.py trainer=cpu
python src/train_diffusion.py trainer=cpu

# train on GPU (default)
python src/train_ae.py
python src/train_diffusion.py
```

You can override any parameter from command line like this

```bash
# train autoencoder on MNIST dataset
python src/train_ae.py datamodule=mnist
# train diffusion model using experiment config
python src/train_diffusion.py model=ddim_vqmodel_f32_celeba
```

## Results


<table>
  <tr>
    <td><img src="assets/origin_mnist.png" alt="Image 1" width="300"><br><p align="center">Original samples from MNIST dataset</p></td>
    <td><img src="assets/synthesis_mnist.png" alt="Image 2" width="300"><br><p align="center">Synthesis samples</p></td>
  </tr>
</table>



<table>
  <tr>
    <td><img src="assets/origin_celeba.png" alt="Image 1" width="300"><br><p align="center">Original samples from CelebA dataset</p></td>
    <td><img src="assets/ddim_vqmodel_f32_celeba.png" alt="Image 2" width="300"><br><p align="center">Synthesis samples</p></td>
  </tr>
</table>
