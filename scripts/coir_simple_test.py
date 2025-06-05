import coir
from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import YourCustomDEModel
import time
all_tests = ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-text2sql","cosqa","codesearchnet","codesearchnet-ccr"]

LIST_model_names = [
# Original model
"intfloat/e5-base-v2",

# COIR models
# *"Salesforce/SFR-Embedding-Code-2B_R",
#"codesage/codesage-large-v2",
# *"Salesforce/SFR-Embedding-Code-400M_R",
#"intfloat/e5-mistral-7b-instruct",
#"Alibaba-NLP/gte-modernbert-base",
#"infly/inf-retriever-v1-1.5b",
# [ ] "infly/inf-retriever-v1",
# [ ] "nvidia/NV-Embed-v1",

# NOT RUN
#"Qodo/Qodo-Embed-1-1.5B",
#"Qodo/Qodo-Embed-1-7B", 

# MTEB models
# *"Salesforce/SFR-Embedding-Mistral",
#"Linq-AI-Research/Linq-Embed-Mistral",
#"jinaai/jina-embeddings-v3",
#"Alibaba-NLP/gte-Qwen2-1.5B-instruct",
#"HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
#"NovaSearch/stella_en_1.5B_v5",

# [ ] "Alibaba-NLP/gte-Qwen2-7B-instruct",
# [ ] "GritLM/GritLM-7B",
# [ ] "nvidia/NV-Embed-v2",
# [ ] "BAAI/bge-m3",

# [ ] "nomic-ai/nomic-embed-code"
# [ ] "nomic-ai/nomic-embed-text-v2-moe"
# [ ] "nomic-ai/nomic-embed-text-v1.5"
# [ ] "sentence-transformers/all-MiniLM-L6-v2"

# NOT RUN
#"gemini-embedding-exp-0307",
#"intfloat/multilingual-e5-large-instruct",
#"Cohere/Cohere-embed-multilingual-v3.0",

]

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", type=str, nargs="+", default=["infly/inf-retriever-v1-1.5b"])
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()

def main():

    args = parse_args()
    model_names = args.model_names
    print(model_names)

    for model_name in model_names:
        # Load the model
        cur = time.time()
        print("="*100)
        print(f"Loading model {model_name}")

        model = YourCustomDEModel(model_name=model_name)

        # Get tasks
        #all task ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-
        # text2sql","cosqa","codesearchnet","codesearchnet-ccr"]
        tasks = get_tasks(all_tests)
        # tasks = get_tasks(tasks=["codetrans-dl"])

        # Initialize evaluation
        evaluation = COIR(tasks=tasks,batch_size=args.batch_size)

        # Run evaluation
        results = evaluation.run(model, output_folder=f"results_coir/{model_name}")
        diff = time.time() - cur
        minutes = diff / 60
        hours = minutes / 60

        print(f"Finished {model_name} in {hours} hours {minutes} minutes")
        print(results)
        print("-"*100)

if __name__ == "__main__":
    main()