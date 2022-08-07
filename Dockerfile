FROM frolvlad/alpine-miniconda3:python3.7

EXPOSE 5000
EXPOSE 8080

# Set the working directory to /malaria_hero
WORKDIR /malaria_hero

# Setup environment with dependences
ADD environment.yml /malaria_hero/
RUN conda env update --name base --file environment.yml \
    && conda clean -afy

# Copy the malaria_hero repo contents into the container at /malaria_hero
ADD . /malaria_hero/

WORKDIR /malaria_hero/src/

ENTRYPOINT ["gunicorn", "--bind", ":8080", "dash_app:server", "--timeout", "180"]

