
import os, sys, json, codecs

def load_all_instances(inpath):
    passages = []
    questions = []
    with codecs.open(inpath, "rU", "utf-8") as f:
        for i, line in enumerate(f):
            inst = json.loads(line.strip())
            inst_id = inst['id']
            ids = []
            content = []
            length = 0
            for j, web_snippet in enumerate(inst['web_snippets']): # for each passage
                p_tok = web_snippet['annotations']['toks'].split()
                ids.append(inst_id+'@%d'%j)
                content.append(p_tok)
                length += len(p_tok)
            #if length > 8000:
            #    thres = 8000/len(ids)
            #    content = [x[:thres] for x in content]
            passages.append(zip(ids, content))
            inst_q_tok = inst['annotations']['toks'].split()
            questions.append((inst_id, inst_q_tok))
    return questions, passages


def collect_vocab(questions, passages, vocab_path):
    vocab = set()
    list_of_vocabs = []
    for qid, q in questions:
        vocab.update(q)
        if len(vocab) > 500000:
            list_of_vocabs.append(vocab)
            vocab = set()

    for inst in passages:
        for pid, p in inst:
            vocab.update(p)
            if len(vocab) > 500000:
                list_of_vocabs.append(vocab)
                vocab = set()

    for v in list_of_vocabs:
        vocab.update(v)

    outfile = open(vocab_path, "wt")
    for word in vocab:
        if len(word) <= 30:
            outfile.write("%s\n" % word.encode('utf-8'))
    outfile.close()
