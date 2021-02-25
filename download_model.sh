#! /bin/bash

# https://drive.google.com/drive/u/1/folders/1zdDWc-5fCgEw82VxMVOWtjgqcPIFA5F4

cwd=$PWD


## ROS Package

echo -e "\e[93m Download hands_only model \e[0m"
cd $cwd/catkin_ws/src/obman_prediction/weight/hands_only
sudo gdown --id 1WnxurAqef3rPAwyHL3lwYflfFLQvKruA
sudo gdown --id 1cJ9HDh9L_bsi7CEUgDH_OOoGXn9X0LYR

echo -e "\e[93m Download mano model \e[0m"
cd $cwd/catkin_ws/src/obman_prediction/weight/mano
sudo gdown --id 1hh3n9SUB_Gp99ECUJ9Db3zVfnbOBQ7ux
sudo gdown --id 1qijDRXxQjn_n_IiTLIO30ZzeU871bmBC
sudo gdown --id 1wVPuDWZSjVPydf2-e1Xr6iAInrAtCH-E
sudo gdown --id 1YtYBvyHpYmsGLFDP9yEII8FFBaQl6ZlB

echo -e "\e[93m Download obman model \e[0m"
cd $cwd/catkin_ws/src/obman_prediction/weight/obman
sudo gdown --id 1lY4hppehnH7o3EeV3RLD8K2It2ndKNHL
sudo gdown --id 1q_nNKqYIju-K3M-GKUBxvUujzo4Jbyib

## assests
echo -e "\e[93m Download contact_zones.pkl \e[0m"
cd $cwd/assets
sudo gdown --id 1VpcJ0J8rA79319AFcphYTq2-fBkeblq6

## misc/mano
echo -e "\e[93m Download mano \e[0m"
cd $cwd/misc/mano
sudo gdown --id 1bzEu0PkjUwfSBF7ZnAZuD9IFD-wGcZBv
sudo gdown --id 1BqIUARRHntGGoKEt4qjCkSWuxDz4NZZH
sudo gdown --id 1W7bcRX8YrVRYjoHfAW55Wmz3IZHHQDre
sudo gdown --id 12Pu7X7_JjoDcqEujDYtEKY6yPZXXkJk4

## obman_dataset
echo -e "\e[93m Download mano_face \e[0m"
cd $cwd/obman_dataset
sudo gdown --id 1Kmhg7UpBQe6-dUwGQ-SsvwdVd8pTys19
sudo gdown --id 1sVDE6RuXqzh55lj1ODkYqqjgqVwCnMkZ

## release_model
echo -e "\e[93m Download release_models \e[0m"
cd $cwd/release_models/fhb
sudo gdown --id 19J5dxeR0Kw-3zQ-PD3BxwyY7BENO17yN
sudo gdown --id 1U6r3_CU9u9JnGYVh2_0qIF6FsvRzKKr4

cd $cwd/release_models/hands_only
sudo gdown --id 1y6RhtRhg9ixhV4KWt2prGv2o9jvqddGq
sudo gdown --id 1lF3T9S249GRt9begKAif0HsTdJSTZW_8

cd $cwd/release_models/obman
sudo gdown --id 1V8mwSGby4qZSuny9TsfAqDU2gNkUQVI0
sudo gdown --id 1_MC6HhcnGByouXx-LioDdbl6PlBiKaMP

cd $cwd