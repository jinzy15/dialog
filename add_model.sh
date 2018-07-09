#!/bin/sh
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
