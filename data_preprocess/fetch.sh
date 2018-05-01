# download dataset and put it in data directory

cd ..
mkdir data
mkdir raw_data
cd ./raw_data

echo "Downloading SimpleQuestions dataset...\n"
wget https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz

echo "\n\nUnzipping SimpleQuestions dataset...\n"
tar -xvzf SimpleQuestions_v2.tgz
rm SimpleQuestions_v2.tgz

cd ./SimpleQuestions_v2
mkdir dataset
mv *.txt ./dataset

cd ..
echo "Downloading FB5M-extra...\n
wget https://www.dropbox.com/s/dt4i1a1wayks43n/FB5M-extra.tar.gz

echo "\n\nUnzipping FB5M-extra...\n"
tar -xzf FB5M-extra.tar.gz
rm FB5M-extra.tar.gz


