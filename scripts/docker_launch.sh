docker build . -t flask_app:0
./docker_allclear.sh
docker run -p 5000:5000 flask_app:0
