<p align="center">
  <img src="https://github.com/caticoa3/malaria_hero/blob/master/images/malaria_hero.jpg?raw=true" alt="Malaria Hero"/ width="430">
</p>

Modified image from the [Center For Infectious Disease Research](https://www.cidresearch.org/blog/human-vs-pathogen-the-art-of-battling-infectious-disease)

## Malaria Hero

Web application that classifies single cells as parasitized or normal; and prioritizes patients based on the percentage of infected cells. More background can be found in this [medium post](https://blog.insightdatascience.com/https-blog-insightdatascience-com-malaria-hero-a47d3d5fc4bb).

Created thanks to publicly available [data from the NIH](https://ceb.nlm.nih.gov/repositories/malaria-datasets/).

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

## Demo
A live demo can be found at [malaria_hero.org](malaria_hero.org)
