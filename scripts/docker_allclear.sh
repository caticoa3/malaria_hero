#!/bin/sh

echo "removing running docker containers..."
docker rm -f $(docker ps -aq)

echo "removing dangling docker images..."
docker rmi $(docker images -qf "dangling=true")

echo "removing dangling volumes..."
docker volume rm $(docker volume ls -qf "dangling=true")
