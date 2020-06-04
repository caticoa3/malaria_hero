<p align="center">
  <img src="https://github.com/caticoa3/malaria_hero/blob/master/images/malaria_hero.jpg?raw=true" alt="Malaria Hero"/>
</p>

Modified image from the Center For Infectous Disease Research
https://www.cidresearch.org/blog/human-vs-pathogen-the-art-of-battling-infectious-disease

## Malaria Hero
https://blog.insightdatascience.com/https-blog-insightdatascience-com-malaria-hero-a47d3d5fc4bb

A web application that classifies single cells as parasitized or normal.

Prioritizing patients based on % of cells infected.


Created thanks to publicly available data: 
https://ceb.nlm.nih.gov/repositories/malaria-datasets/

## Running locally 
### Option 1: in a python enviroment
use the provided environment.yml to create the conda environment and run the following
    cd src/
    python dash_app.py 

### Option 2: in a docker swarm
    cd src/
    make build # to build the needed containers and environment
    make deploy # to run

in both cases you can navigate to 0.0.0.0:5000 in your favorite flavor of internet browser