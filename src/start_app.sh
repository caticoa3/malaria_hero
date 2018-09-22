#!/bin/bash

NAME="Malarias_Screening-App"
NUM_WORKERS=4
TIMEOUT=1200

echo "Starting $NAME"

# Create the run directory if it doesn't exist
# RUNDIR=$(dirname $SOCKFILE)

export >> env.log

loglevel="debug"

if [ $PROD ];then
   loglevel="info"
fi

# Start your unicorn
exec gunicorn flask_app:app -b 0.0.0.0:5000 \
  --name $NAME \
  --timeout $TIMEOUT \
  --workers $NUM_WORKERS \
  --log-level=$loglevel \

  # --bind=unix:$SOCKFILE

###  if using serverless
# python app.py
