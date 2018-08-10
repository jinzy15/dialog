#!/bin/sh
mkdir $1
cd $1
rm -rf data
rm -rf transformer
rm -rf dataset
rm -rf dataprocess
rm -rf utils
ln -s ../data data
ln -s ../transformer transformer
ln -s ../dataset dataset
ln -s ../dataprocess dataprocess
ln -s ../utils utils
mkdir train_fruit
touch main.py
touch Config.py

