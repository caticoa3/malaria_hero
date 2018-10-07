#!/bin/sh

docker swarm init
docker network create --driver=overlay --attachable mynetwork
docker stack deploy -c docker-compose.yml malaria_hero
