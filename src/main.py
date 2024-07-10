# Imports
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
from .utils.retriever import mean_pooling
from .utils.rag import format_prompt
from vllm import LLM, SamplingParams

# TODO: Cli

query = "Tell me about the first person to climb the Himalaya"
# Dataset Loading 
## Datasets 
### ASQA 
# asqa = load_dataset("din0s/asqa", split='dev')
### Biography writing
### 

# Model Loading
## There should be 4 main models for prompt creation:
### - A Summarizer model. Choice so far: Self-RAG 
rag_model = LLM("selfrag/selfrag_llama2_7b", download_dir="/netscratch/fonseca/hf_cache", dtype="half")
rag_sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)


### - A General Purpose Model 
#-----------------------------------------------------------------------#
### - A Retriever - Contriever
### TODO: Create a function to receive the query and return the documents from contriever_model
### The main code is extremely confusing at this point
# contriver_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
# contriever_model = AutoModel.from_pretrained('facebook/contriever')
# 
# # Apply tokenizer
# inputs = contriver_tokenizer(query, padding=True, truncation=True, return_tensors='pt')
# 
# # Compute token embeddings
# outputs = contriever_model(**inputs)
# query_embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
# 
# 
# # Hardcoded Paths - #TODO add cli
# # defaults from original code
# projection_size = 768 
# n_subquantizers = 0 
# n_bits = 8
# passages_embeddings = "/netscratch/fonseca/thesis_stuff/retriever/wikipedia_embeddings/*"
# save_or_load_index = True
# indexing_batch_size = 1000000
# passages = "/netscratch/fonseca/thesis_stuff/retriever/psgs_w100.tsv"
# 
# # index = src.index.Indexer(projection_size, n_subquantizers, n_bits )
# 
# # index all passages
# input_paths = glob.glob(passages_embeddings)
# input_paths = sorted(input_paths)
# embeddings_dir = os.path.dirname(input_paths[0])
# index_path = os.path.join(embeddings_dir, "index.faiss")
# if save_or_load_index and os.path.exists(index_path):
#     index.deserialize_from(embeddings_dir)
# else:
#     print(f"Indexing passages from files {input_paths}")
#     start_time_indexing = time.time()
#     index_encoded_data(index, input_paths, indexing_batch_size)
#     print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
#     if save_or_load_index:
#         index.serialize(embeddings_dir)
# 
#     # load passages
#     passages = src.data.load_passages(args.passages)
#     passage_id_map = {x["id"]: x for x in passages}
# 
# top_ids_and_scores = index.search_knn(query_embeddings, n_docs=1)

# retrieve top_k and find them on index map


###

# Score between docs is calculated with the dot product between them
# score01 = embeddings[0] @ embeddings[1] #1.0473

# -------------------------------------------------------------------------#
## For the output:
### - A General summarizer model (Llama 13B?)
llama_tokenizer = AutoTokenizer.from_pretrained('')
llama_model = AutoModel.from_pretrained('')

# Prompt
## get it from dataset