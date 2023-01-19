import sys, csv
from transformers import BertTokenizer

if len(sys.argv) != 2:
    print("Usage: python3 %s file" % sys.argv[0])
    exit()

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
with open(sys.argv[1] + '.rationale_idx', 'w') as f_idx:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in lines:
        line = line.split('\t')
        label = line[0].strip()
        sent1 = line[1].strip()
        sent2 = line[2].strip()

        rationale_idx = ['0' for _ in range(len(tokenizer.convert_ids_to_tokens(tokenizer.batch_encode_plus([(sent1, sent2)])['input_ids'][0])))]

        if len(rationale_idx) != len(tokenizer.convert_ids_to_tokens(tokenizer.batch_encode_plus([(sent1, sent2)])['input_ids'][0])):
            print(line)
            import pdb; pdb.set_trace()
        f_idx.write(' '.join(rationale_idx) + '\n')