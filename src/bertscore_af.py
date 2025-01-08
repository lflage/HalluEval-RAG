import sys
import os
import json
import pandas as pd
from transformers import BertTokenizer, BertModel
from bert_score import BERTScorer

data_path = "/netscratch/fonseca/HalluEval-RAG/data/atomic_facts/InstructGPT_llama_annotations.jsonl"
out_file = os.path.basename(data_path).replace(".jsonl","bertscores.csv")
out_folder = "../results/metrics/afg_bert_score/"
out_path = os.path.join(out_folder, out_file)

# Load BERTScorer
scorer = BERTScorer(model_type='bert-base-uncased')


# Create lists to send to dataframe
## columns: doc_id , sentence, af, bert_score
doc_id = []
sentences = []
afs = []
precision = []
recall = []
f1 = []

with open(data_path, "r", encoding="utf-8") as f:
    for i, json_f in enumerate(f):
        data_dict = json.loads(json_f)
        for annotation in data_dict['annotations']:
            ref = annotation['text']
            for atomic_fact in annotation["llama-atomic-facts"]:
                cur_af = atomic_fact['text']
                cur_doc = i

                # BertScore calculation
                P, R, F = scorer.score([cur_af], [ref])

                doc_id.append(i)
                sentences.append(ref)
                afs.append(cur_af)
                precision.append(P[0].item())
                recall.append(R[0].item())
                f1.append(F[0].item())



assert len(doc_id)==len(sentences)==len(afs)==len(precision)

df = pd.DataFrame(data={"doc": doc_id,
                        "sentence": sentences,
                        "atomic_fact": afs,
                        "precision":precision,
                        "recall": recall,
                        "f1-score": f1})

print(df.head())

df.to_csv(out_path)

# Aggregating the data for results
