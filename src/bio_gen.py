import os
import sys
import json
import argparse
import torch
import factscore
from tqdm import tqdm
from factscore import factscorer
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from utils import ROOT, SYSTEM_INSTRUCT, get_docs_asqa, summarize_docs

os.chdir(ROOT)

parser = argparse.ArgumentParser(
    description="CLI for configuring the setups for Biography writing task experiments.")

parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Model name or path")
parser.add_argument("--out-dir", type=str, default="./results/bio",
                    help="Output directory for results")
parser.add_argument("--dataset-path", type=str, default="./data/datasets/bio_queries.jsonl",
                    help="Path to the dataset file")
parser.add_argument("--is-rag", action="store_true",
                    help="Enable RAG mode (default: False, set flag to enable)")
parser.add_argument('--top_k', type=int, default=3,
                    help="Number of top documents to consider for RAG prompting")
parser.add_argument('--is-summ', required='--is-rag' in sys.argv, action="store_true",
                    help="Enable summmarization for RAG docs")

args = parser.parse_args()

MODEL_NAME = args.model_name
OUT_DIR = args.out_dir
DATASET_PATH = args.dataset_path
IS_RAG = args.is_rag
IS_SUMM = args.is_summ

model_base_name = MODEL_NAME.rsplit("/", maxsplit=1)[-1]
OUT_FILE = f"bios_{model_base_name}_is_rag={IS_RAG}_is_summ={IS_SUMM}.jsonl"
OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)

with open (DATASET_PATH, "r", encoding="utf8") as f:
    bios = [json.loads(line) for line in f]

# Model Init
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.eos_token_id

use_chat = "chat_template" in tokenizer.init_kwargs

model = model.to(DEVICE)

print("Init model reponse")
model_responses = []
for sample in tqdm(bios):
    topic = sample['input'].split(":")[-1].strip()
    if IS_RAG:
        docs = get_docs_asqa(sample, 3)

        if IS_SUMM:
            docs = summarize_docs(model, tokenizer, docs, DEVICE)

        question = f"""Answer the following question based on the given documents.
        Documents: {docs}
        Question: {sample['input']}
        Answer
        """
    else:
        question = sample['input']

    if use_chat:
        instruct_dict = [{"role" : "system", "content": SYSTEM_INSTRUCT},
                  {"role": "user", "content": question}]
        question = tokenizer.apply_chat_template(instruct_dict,
                                                tokenize=False,
                                                add_generation_prompt=True)

    tokens = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    cur_iids = tokens.input_ids.to(DEVICE)
    attention_mask = tokens.attention_mask.to(DEVICE)   # Ensure attention mask is properly handled

    out = model.generate(
        cur_iids,
        attention_mask=attention_mask,                  # Pass the attention mask
        pad_token_id=tokenizer.eos_token_id,            # Set pad_token_id to eos_token_id
        return_dict_in_generate=True,
        max_new_tokens=256
    )

    gen_tokens = out["sequences"]
    out_text = tokenizer.decode(
        gen_tokens[0, cur_iids.shape[-1]:],
        skip_special_tokens=True
    )

    sample_out = {
        "topic"     :   topic,
        "input"     :   sample['input'],
        "prompt"    :   question,
        "rag"       :   IS_RAG,
        "output"    :   out_text,
    }

    model_responses.append(sample_out)


# Write model responses
## Json Format:
## {"topic"     : [list of topics],
##  "input"     : str,
##  "output"    : str (model response),
##  "rag"       : bool
#   }

print(f"\n Saving to {OUT_FILE}")
with open(OUT_FILE, "w", encoding='utf8') as f:
    for sample in model_responses:
        f.write(json.dumps(sample)+"\n")
