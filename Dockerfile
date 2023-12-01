FROM 12.0.0-devel-ubuntu22.04

WORKDIR /lung-diffusion

COPY ./requirements.txt /lung-diffusion/requirements.txt

RUN pip install -r requirements.txt

COPY . /lung-diffusion



