#! /bin/bash

# https://drive.google.com/drive/u/1/folders/1N-G67bdqoh1aiJEECjXYLYJwBcp753cE
sudo apt update
sudo apt install unzip

cwd=$PWD

echo -e "\e[93m Download obman \e[0m"
cd $cwd/datasymlinks
sudo gdown --id 1J0XpRbNKAryFJxGGVEQ4dsIAmSOEMcEu
sudo unzip obman.zip


echo -e "\e[93m Download ShapeNetCore.v2 \e[0m"
cd $cwd/datasymlinks
sudo gdown --id 1wtGdlUw-H8ndDWpCzSgwWY9anYcv1HZn
sudo unzip ShapeNetCore.v2.zip

cd $cwd