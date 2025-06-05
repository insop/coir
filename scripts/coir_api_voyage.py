import time
import os
import coir
from coir.data_loader import get_tasks
from coir.evaluation import COIR
import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
from tqdm.auto import tqdm
import voyageai
from dotenv import load_dotenv
load_dotenv()

all_tests = ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-text2sql","cosqa","codesearchnet","codesearchnet-ccr"]

provider = "voyage"
model_names = ["voyage-3.5", "voyage-3.5-lite"]
# model_names = ["voyage-3-large", "voyage-3"]
# model_names = [ "voyage-code-3"]
# model_names = ["voyage-3-lite", "voyage-code-3"]
# model_names = ["voyage-3", "voyage-3-lite", "voyage-code-3"]
# # model_names = ["voyage-3-large", "voyage-3", "voyage-3-lite", "voyage-code-3"]

class APIModel:
    def __init__(self, model_name="voyage-code-3", **kwargs):
        # Initialize the voyageai client
        api_key = os.getenv("VOYAGE_API_KEY")
        self.vo = voyageai.Client(api_key=api_key)  # This uses VOYAGE_API_KEY from environment
        self.model_name = model_name
        self.requests_per_minute = 30  # Max requests per minute
        self.delay_between_requests = 60 / self.requests_per_minute  # Delay in seco

    def encode_text(self, texts: list, batch_size: int = 12, input_type: str = "document") -> np.ndarray:
        logging.info(f"Encoding {len(texts)} texts...")

        all_embeddings = []
        start_time = time.time()
        # Processing texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            result = self.vo.embed(batch_texts, model=self.model_name, input_type=input_type,truncation=True)
            batch_embeddings = result.embeddings  # Assume the API directly returns embeddings
            all_embeddings.extend(batch_embeddings)
            # Ensure we do not exceed rate limits
            time_elapsed = time.time() - start_time
            if time_elapsed < self.delay_between_requests:
                time.sleep(self.delay_between_requests - time_elapsed)
                start_time = time.time()

        # Combine all embeddings into a single numpy array
        embeddings_array = np.array(all_embeddings)

        # Logging after encoding
        if embeddings_array.size == 0:
            logging.error("No embeddings received.")
        else:
            logging.info(f"Encoded {len(embeddings_array)} embeddings.")

        return embeddings_array

    def encode_queries(self, queries: list, batch_size: int = 12, **kwargs) -> np.ndarray:
        truncated_queries = [query[:256] for query in queries]
        truncated_queries = ["query: " + query for query in truncated_queries]
        query_embeddings = self.encode_text(truncated_queries, batch_size, input_type="query")
        return query_embeddings


    def encode_corpus(self, corpus: list, batch_size: int = 12, **kwargs) -> np.ndarray:
        texts = [doc['text'][:512]  for doc in corpus]
        texts = ["passage: " + doc for doc in texts]
        return self.encode_text(texts, batch_size, input_type="document")

def main():
    # Load the model
    for model_name in model_names:
        model = APIModel(model_name=model_name)

        cur = time.time()
        print("="*100)
        print(f"Model {model_name}")

        # Get tasks
        tasks = coir.get_tasks(all_tests)

        # Initialize evaluation
        evaluation = COIR(tasks=tasks, batch_size=64)

        # Run evaluation
        results = evaluation.run(model, output_folder=f"results_coir/{provider}/{model_name}")
        diff = time.time() - cur
        minutes = diff / 60
        hours = minutes / 60

        print(f"Finished {model_name} in {hours} hours {minutes} minutes")
        print(results)
        print("-"*100)

if __name__ == "__main__":
    main()