from transformers import AutoTokenizer,LlamaForCausalLM, pipeline
from datasets import load_dataset
import json
import torch
from tqdm import tqdm
import os
import sys


os.chdir(sys.path[0])

def make_prompt_rag(top_k, row):
    concat_docs = ""
    for a in row['ctxs'][:top_k]:
        concat_docs += a['text']
        concat_docs +=' '
    prompt = """[INST]Answer the following question based on the given documents:\n
    Question: {}\n
    Documents: {}[/INST]
    """.format( row['input'],
                concat_docs)
    return prompt

def make_bio_prompt(row):

    prompt = "[INST]{}[/INST]".format(row['input'])
    return prompt

# TODO: Make CLI
DATA_FILES = "../datasets/bio_queries.jsonl"
out_file = os.path.basename(DATA_FILES)
OUTPUT_FILE = os.path.join("../results","rag_"+out_file)
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load datasets
dataset = load_dataset('json', data_files=DATA_FILES)

print("loaded dataset")
# load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(MODEL_NAME).to(device)
print("loaded model")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device=device,
    batch_size=8
)

print("created pipeline")

list_of_reponses = []
for sample in tqdm(dataset['train']):
    query = make_prompt_rag(3, sample)
    
    response = pipe(query,
        do_sample=True,
        top_k=3,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    response_start_ix = len(query)
    response_string = response[0]['generated_text']

    # Create an output dictionary
    output = {
        'query': query,
        'response': response_string[response_start_ix:]
    }
    list_of_reponses.append(output)

with open(OUTPUT_FILE, "w") as f:
    for res in list_of_reponses:
        json.dump(res,f)
        f.write("\n")