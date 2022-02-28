#!/bin/bash

if [[ -z $1 ]]; then
  echo "Please entry name file with python code after command. For example ./start.sh 1.py";
  exit 0;
fi



echo "Running docker environment with dependencies"
docker run --rm -v $(pwd):/mnt ravino/devops:ml $1
echo "Stopped docker environments"
