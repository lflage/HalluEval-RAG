# Script para gerar o FactScore de produções de certos modelos
# Definições:


# TODO: CLI
# CLI: deve conter:
#   - caminho de saída, documento deve conter FactScores individuais
#   - Nome do modelo
#   -

from factscore.factscorer import FactScorer
import json
fs = FactScorer(openai_key="")


# topics: list of strings (human entities used to generate bios)
# generations: list of strings (model generations)
topics = ["Ludwig van Beethoven",
            "Abraham Lincoln",
            "Luiz Inácio Lula da Silva",
            "Olaf Scholz",
]
generations = []
with open("../results/bio/bio_writing_test.jsonl", "r") as f:
    for line in f.readlines():
        cur_dict = json.load(line)
        generations.append(cur_dict['generated_text'])

out = fs.get_score(topics, generations, gamma=10)
print (out["score"]) # FActScore
print (out["init_score"]) # FActScore w/o length penalty
print (out["respond_ratio"]) # % of responding (not abstaining from answering)
print (out["num_facts_per_response"]) # average number of atomic facts per response