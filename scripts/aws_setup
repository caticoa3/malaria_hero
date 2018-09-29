sudo yum update -y
sudo yum install git
git clone 
#transfer trained models
scp -i ~carlos/Desktop/AWS\ self\ guided\ tutorial/learning.pem -r models/ ec2-user@ec2-18-188-29-9.us-east-2.compute.amazonaws.com:~/malaria/models

sudo yum install docker -y
sudo service start docker
sudo usermod -a -G docker ec2-user

#install docker-compose
pip install docker-compose

#follow with 
docker-compose build

cd src/
docker-compose up

#access port 5000
