from transformers import AutoTokenizer,LlamaForCausalLM, pipeline
from datasets import load_dataset
import json
import torch
from tqdm import tqdm
import os
import sys
import argparse


os.chdir(sys.path[0])

def parse_args():
    parser = argparse.ArgumentParser(description="Run a LLaMA RAG/Bio pipeline on a dataset")
    
    parser.add_argument('--data_file', type=str, default="../datasets/llama2Chat_bio_prompts_ctxs.jsonl", help="Path to the input JSONL dataset file")
    parser.add_argument('--output_dir', type=str, default="../results", help="Directory to save the output file")
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Name of the pretrained model to use")
    parser.add_argument('--rag', action='store_true', help="Use RAG-style prompt creation if set")
    parser.add_argument('--top_k', type=int, default=3, help="Number of top documents to consider for RAG prompting")

    return parser.parse_args()

def make_prompt_rag(top_k, row):
    concat_docs = ""
    for a in row['ctxs'][:top_k]:
        concat_docs += a['text']
        concat_docs +=' '
    prompt = """[INST]Answer the following question based on the given documents:\n
    {}\n
    Documents: {}[/INST]
    """.format( row['input'],
                concat_docs)
    return prompt

def make_bio_prompt(row):

    prompt = "[INST]{}[/INST]".format(row['input'])
    return prompt

if __name__=="__main__":
    args = parse_args()
    
    DATA_FILES = args.data_file
    out_file = os.path.basename(DATA_FILES)
    MODEL_NAME = args.model_name
    RAG = args.rag
    top_k = args.top_k

    # Set appropriate output file based on RAG option
    if RAG:
        OUTPUT_FILE = os.path.join(args.output_dir, "rag_results_" + out_file)
        make_prompts = make_prompt_rag
    else:
        OUTPUT_FILE = os.path.join(args.output_dir, "results_" + out_file)
        make_prompts = make_bio_prompt

    print(f"RAG: {RAG}")
    print(f"Data file: {DATA_FILES}")
    print(f"Model name: {MODEL_NAME}")
    print(f"Output file: {OUTPUT_FILE}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load datasets
    dataset = load_dataset('json', data_files=DATA_FILES)

    print("loaded dataset")
    # load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME).to(device)

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
        # Create prompt based on RAG or BIO setting
        if RAG:
            query = make_prompt_rag(top_k, sample)
        else:
            query = make_bio_prompt(sample)
        
        response = pipe(query,
            do_sample=True,
            top_k=3,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        response_start_ix = len(query)
        response_string = response[0]['generated_text']

        # Create an output dictionary
        sample['output'] =  response_string[response_start_ix:]
        list_of_reponses.append(sample)

    with open(OUTPUT_FILE, "w") as f:
        for res in list_of_reponses:
            json.dump(res,f)
            f.write("\n")