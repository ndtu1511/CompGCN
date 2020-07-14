from case_study.utils import load_data, get_label
import torch
import json
import os
import pandas as pd


class Knowledge_Graph:
    def __init__(self, path_name, model, device):
        self.ent2id, self.id2ent, \
        self.rel2id, self.id2rel, \
        self.data, self.sr2o_test, self.label_test = load_data(path_name)
        with open(os.path.join('data', path_name, 'ent2string.json')) as f:
            self.ent_string = json.load(f)
        self.model = model
        self.device = device
        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id) // 2

        un_string_ent_list = list(set(self.ent2id.keys()) - set(self.ent_string.keys()))
        self.un_string_ent_list = [self.ent2id[k] for k in un_string_ent_list]

    def link_prediction(self, entity, relation, tail_predict, n, threshold):
        ent = self.ent2id[entity]
        rel = self.rel2id[relation]
        if not tail_predict:
            rel = self.num_rel + rel
        label_in_test_dataset = self.label_test[(ent, rel)]

        ent = torch.LongTensor([ent], device=self.device)
        rel = torch.LongTensor([rel], device=self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model.predict(ent, rel)
            try:
                label = self.sr2o_test[(ent.item(), rel.item())]
                label += self.un_string_ent_list
                label = get_label(label, self.num_ent, self.device)
            except:
                label = get_label(self.un_string_ent_list, self.num_ent, self.device)
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred).squeeze()
            sorted, indices = torch.sort(pred, descending=True)
            sorted = sorted.numpy()
            indices = indices.numpy()
            predicted_ent = indices[:n] if n is not None else indices[sorted >= threshold]
            results = []

            for i, idx in enumerate(predicted_ent):
                existing_in_test_dataset = True if idx in label_in_test_dataset else False
                if tail_predict:
                    head = self.ent_string[entity]
                    tail = self.ent_string[self.id2ent[idx]]
                else:
                    head = self.ent_string[self.id2ent[idx]]
                    tail = self.ent_string[entity]
                results.append([
                    head,
                    relation,
                    tail,
                    str(round(sorted[i], 3)),
                    i + 1,
                    existing_in_test_dataset
                ])

            result_df = pd.DataFrame(results,
                                     columns=['HEAD', 'RELATION', 'TAIL', 'SCORE', 'RANK', 'EXISTING_IN_TEST_DATASET'])
            return result_df
