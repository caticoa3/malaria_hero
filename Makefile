deploy:

	docker swarm init
	docker network create --driver=overlay --attachable mynetwork
	docker stack deploy -c docker-compose.yml malaria_hero

docker_allclear:

	@echo "take malaria_hero server down..."
	docker stack rm malaria_hero

	@echo "removing running docker containers..."
	docker rm -f $(docker ps -aq)

	@echo "removing dangling docker images..."
	docker rmi $(docker images -qf "dangling=true")

	@echo "removing dangling volumes..."
	docker volume rm $(docker volume ls -qf "dangling=true")

clear:

	@echo "clearing running docker containers without removing services..."

	@echo "removing running docker containers..."
	docker rm -f $(docker ps -aq)

	@echo "removing dangling docker images..."
	docker rmi $(docker images -qf "dangling=true")

	@echo "removing dangling volumes..."
	docker volume rm $(docker volume ls -qf "dangling=true")

docker_launch:

	docker build . -t flask_app:0
	make docker_allclear
	docker run -p 5000:5000 flask_app:0

update_all_submodules:

	git submodule init # Ensure the submodule points to the right place
	git submodule sync    # Ensure the submodule points to the right place
	git submodule update  # Update the submodule  
	git submodule foreach git checkout master  # Ensure subs are on master branch
	git submodule foreach git pull origin master # Pull the latest master