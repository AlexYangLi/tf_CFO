#!/usr/bin/env bash
# download dataset and put it in data directory

cd ..
mkdir data
mkdir raw_data
cd ./raw_data

printf "\nDownloading SimpleQuestions dataset...\n"
#wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz

printf "\nUnzipping SimpleQuestions dataset...\n"
tar -xvzf SimpleQuestions_v2.tgz
rm SimpleQuestions_v2.tgz

cd ./SimpleQuestions_v2
mkdir dataset
mv annotated*.txt ./dataset

cd ..
printf "\nDownloading FB5M-extra...\n"
#wget https://www.dropbox.com/s/dt4i1a1wayks43n/FB5M-extra.tar.gz

echo "\nUnzipping FB5M-extra...\n"
tar -xzf FB5M-extra.tar.gz
rm FB5M-extra.tar.gz
mkdir FB5M-extra
mv FB5M.en-name.txt FB5M-extra/
mv FB5M.name.txt FB5M-extra/
mv FB5M.type.txt FB5M-extra/


