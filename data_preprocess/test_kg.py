from utils import www2fb, pickle_load

triples = pickle_load('../data/FB5M_triple.pkl')
entities = pickle_load('../data/FB5M_idx2entity.pkl')
relations = pickle_load('../data/FB5M_idx2relation.pkl')

with open('../raw_data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt', 'r')as reader:
    for index, line in enumerate(reader):
        if index % 10000 == 0:
            print(index)

        fields = line.strip().split('\t')
        subject = www2fb(fields[0])
        relation = www2fb(fields[1])
        objects = fields[2].split()

        if subject not in entities or relation not in relations or subject not in triples.keys():
            print(index, line)
            exit(1)

        for obj in objects:
            if www2fb(obj) not in entities:
                print(index, line)
                exit(1)

