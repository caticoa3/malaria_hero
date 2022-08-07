setup_instance:

	sudo yum update -y
	sudo yum install git
	git clone https://github.com/caticoa3/malaria_hero.git 

	sudo yum install docker -y
	sudo service start docker
	sudo usermod -a -G docker ec2-user
	pip install docker-compose

build: 

	docker-compose build 
	docker rmi nginx:1.15.2
	docker rmi frolvlad/alpine-miniconda3:python3.7

server_build:

	# pull from atico/malaria_hero_api:tf_lite on hub.docker.com
	docker-compose -f onserver.docker-compose.yml build
	docker rmi nginx:1.15.2
	docker rmi atico/malaria_hero_api:tf_lite

push_image:

	# pushing takes a long time, its better to let the image be built in dockerhub
	docker tag malaria_hero_api atico/malaria_hero_api:tflite
	docker push atico/malaria_hero_api:tflite

deploy:

	docker swarm init
	docker network create --driver=overlay --attachable mynetwork
	docker stack deploy -c docker-compose.yml malaria_hero

update_and_deploy:

	make allclear
	make build
	make deploy

allclear:

	@echo "take malaria_hero server down..."
	docker stack rm malaria_hero

	@echo "removing running docker containers..."
	docker rm -f $$(docker ps -aq)

	docker swarm leave -f

	@echo "removing dangling docker images..."
	docker image prune -f 

	@echo "removing dangling volumes..."
	docker volume prune -f 

clear:

	@echo "clearing running docker containers without removing services..."

	@echo "removing running docker containers..."
	docker rm -f $$(docker ps -aq)

	@echo "removing dangling docker images..."
	docker rmi $$(docker images -qf "dangling=true")

	@echo "removing dangling volumes..."
	docker volume rm $$(docker volume ls -qf "dangling=true")

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

view_log:

	docker logs $$(docker ps -qf "name=malaria_hero_api") -f

browse_files:

	docker exec -t -i $$(docker ps -qf "name=malaria_hero_api") /bin/sh

destroy:

	docker rmi $$(docker images -qf "reference=malaria*")

deploy_gcp:

	docker build -f Dockerfile -t gcr.io/malaria-hero/mh_api:0 .
	docker push gcr.io/malaria-hero/mh_api:0
	gcloud run deploy malaria-hero --image=gcr.io/malaria-hero/mh_api:0 --platform=managed --region=us-west1 ;\
	--timeout=60 --concurrency=5 --cpu=2 --memory=1024Mi --max-instances=4 --allow-unauthenticated 
