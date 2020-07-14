from ordered_set import OrderedSet
from collections import defaultdict as ddict
import numpy as np
import torch

np.set_printoptions(precision=4)
def load_data(dataset):
    ent_set, rel_set = OrderedSet(), OrderedSet()
    for split in ['train', 'test', 'valid']:
        for line in open('./data/{}/{}.txt'.format(dataset, split)):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            ent_set.add(sub)
            rel_set.add(rel)
            ent_set.add(obj)

    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    rel2id.update({rel+'_reverse': idx+len(rel2id)
                   for idx, rel in enumerate(rel_set)})

    id2ent = {idx: ent for ent, idx in ent2id.items()}
    id2rel = {idx: rel for rel, idx in rel2id.items()}

    num_ent = len(ent2id)
    num_rel = len(rel2id) // 2

    data = ddict(list)
    sr2o = ddict(set)

    for split in ['train', 'test', 'valid']:
        for line in open('./data/{}/{}.txt'.format(dataset, split)):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            sub, rel, obj = ent2id[sub], rel2id[rel], ent2id[obj]
            data[split].append((sub, rel, obj))

            if split == 'train':
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel+num_rel)].add(sub)

    data = dict(data)
    sr2o_n = {k: list(v) for k, v in sr2o.items()}

    for sub, rel, obj in data['valid']:
        sr2o[(sub, rel)].add(obj)
        sr2o[(obj, rel+num_rel)].add(sub)
    sr2o_test = {k: list(v) for k, v in sr2o.items()}
    label_test = ddict(set)
    for sub, rel, obj in data['test']:
        sr2o[(sub, rel)].add(obj)
        sr2o[(obj, rel+num_rel)].add(sub)
        label_test[(sub, rel)].add(obj)
        label_test[(obj, rel+num_rel)].add(sub)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}
    triples = ddict(list)

    for (sub, rel), obj in sr2o_n.items():
        triples['train'].append(
            {'triple': (sub, rel, -1), 'label': sr2o_n[(sub, rel)], 'sub_samp': 1})

    for split in ['test', 'valid']:
        for sub, rel, obj in data[split]:
            rel_inv = rel + num_rel
            triples['{}_{}'.format(split, 'tail')].append(
                {'triple': (sub, rel, obj), 	   'label': sr2o_all[(sub, rel)]})
            triples['{}_{}'.format(split, 'head')].append(
                {'triple': (obj, rel_inv, sub), 'label': sr2o_all[(obj, rel_inv)]})
    return ent2id, id2ent, rel2id, id2rel, data, sr2o_test, label_test

def get_label(label, num_ent, device):
    y = np.zeros([num_ent], dtype=np.float32)
    if label!=False:
        for e2 in label: y[e2] = 1.0
    return torch.FloatTensor(y, device=device)