#! /bin/bash

# https://drive.google.com/drive/u/1/folders/1zdDWc-5fCgEw82VxMVOWtjgqcPIFA5F4

cwd=$PWD

echo -e "\e[93m Download hands_only model \e[0m"
cd $cwd/weight/hands_only
sudo gdown --id 1WnxurAqef3rPAwyHL3lwYflfFLQvKruA
sudo gdown --id 1cJ9HDh9L_bsi7CEUgDH_OOoGXn9X0LYR

echo -e "\e[93m Download mano model \e[0m"
cd $cwd/weight/mano
sudo gdown --id 1hh3n9SUB_Gp99ECUJ9Db3zVfnbOBQ7ux
sudo gdown --id 1qijDRXxQjn_n_IiTLIO30ZzeU871bmBC
sudo gdown --id 1wVPuDWZSjVPydf2-e1Xr6iAInrAtCH-E
sudo gdown --id 1YtYBvyHpYmsGLFDP9yEII8FFBaQl6ZlB

echo -e "\e[93m Download obman model \e[0m"
cd $cwd/weight/obman
sudo gdown --id 1lY4hppehnH7o3EeV3RLD8K2It2ndKNHL
sudo gdown --id 1q_nNKqYIju-K3M-GKUBxvUujzo4Jbyib
