echo "\n\n create knowledge graph\n"
python create_kg.py ../raw_data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt ../raw_data/SimpleQuestions_v2/dataset/ ../data

echo "\n\n create mapping: subject to name\n"
python create_index_names.py ../raw_data/FB5M-extra/FB5M.name.txt ../data

echo "\n\n create mapping: subject to type\n"
python create_index_type.py ../raw_data/FB5M-extra/FB5M.type.txt ../data

echo "\n\n trim name and type for subject\n"
python trim_name_and_type.py ../data/FB5M_entity2name.pkl ../data/FB5M_entity2type.pkl ../data/FB5M_triple.pkl ../data

echo "\n\ncreate mapping: name to subject and ngram to subject\n"
python create_ngram2entity.py ../data/trim_subject2name.pkl ../data

echo "\n\npretrain word embeddings"
python pretrain_embedding.py ../raw_data/SimpleQuestions_v2/dataset/ ../data

echo "\n\ngenerate train/valid/test data"
python preprocess_dataset.py ../raw_data/SimpleQuestions_v2/dataset/ ../data/trim_subject2name.pkl ../data/trim_subject2type.pkl ../data
