import sys, csv
from transformers import BertTokenizer

if len(sys.argv) != 2:
    print("Usage: python3 %s file" % sys.argv[0])
    exit()

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
with open(sys.argv[1] + '.rationale_idx', 'w') as f_idx, open(sys.argv[1] + '.rationale_token', 'w') as f_token:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in lines:
        line = line.split('\t')
        label = line[0].strip()
        sent1 = line[1].strip()
        sent2 = line[2].strip()
        sent1_marked = line[3].strip()
        sent2_marked = line[4].strip()
        sent1_marked = sent1_marked.split('*')
        sent2_marked = sent2_marked.split('*')
        rationale_token, rationale_idx = [], ['0']
        tokens = ['[CLS]']
        for sent in [sent1_marked, sent2_marked]:
            for i, word in enumerate(sent):
                token = tokenizer.tokenize(word)
                tokens += token
                if i % 2 == 0:
                    rationale_idx += ['0' for _ in range(len(token))]
                else:
                    if token[-1] == ',':
                        rationale_idx += ['1' for _ in range(len(token) - 1)]
                        rationale_idx += ['0']
                        rationale_token += token[:-1]
                    else:
                        rationale_idx += ['1' for _ in range(len(token))]
                        rationale_token += token
            rationale_token += ['|']
            tokens += ['[SEP]']
            rationale_idx += ['0']

        if len(rationale_idx) != len(tokenizer.convert_ids_to_tokens(tokenizer.batch_encode_plus([(sent1, sent2)])['input_ids'][0])):
            print(line)
            import pdb; pdb.set_trace()
        f_idx.write(' '.join(rationale_idx) + '\n')
        f_token.write(' '.join(rationale_token[:-1]) + '\n')