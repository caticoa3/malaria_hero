FROM continuumio/miniconda3:latest

EXPOSE 5000

# Install updates to Ubuntu base if any 
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Add the user that will run the app (no need to run as root)
#RUN groupadd -r myuser && useradd -r -g myuser myuser

# Set the working directory to /malaria_hero
WORKDIR /malaria_hero

# Setup environment with dependences
ADD requirements_conda.txt /malaria_hero/
RUN conda update conda
RUN conda config --add channels conda-forge
RUN conda config --append channels menpo
RUN conda install --yes --quiet --file requirements_conda.txt
RUN conda clean --yes --tarballs

#Copy the $INSIGHT contents into the container at /malaria_hero
ADD . /malaria_hero/

ENV NAME malaria_hero

WORKDIR /malaria_hero/src/

#CMD python dash_table.py
#CMD python flask_app.py
