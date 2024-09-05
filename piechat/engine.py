__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import re
from datetime import datetime

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .config import GlobalConfig


class PieChat:
    def __init__(self, config: GlobalConfig):
        llm_args = AsyncEngineArgs(
            model=str(config.llm_config.llm_path),
            max_model_len=config.llm_config.max_model_length,
            dtype=config.llm_config.dtype,
            gpu_memory_utilization=config.llm_config.gpu_memory_utilization,
            tensor_parallel_size=config.llm_config.tensor_parallel_size
        )
        self.llm = AsyncLLMEngine.from_engine_args(llm_args)
        self.llm_config = config.llm_config

        self.embedding = HuggingFaceEmbeddings(
            model_name=str(config.emb_config.embedding_path),  # Does not accept Path
            model_kwargs={
                "device": config.emb_config.emb_device,
                "trust_remote_code": not config.emb_config.no_trust_remote_code,
                "model_kwargs": {
                    "attn_implementation": config.emb_config.attn_implementation,
                    "torch_dtype": getattr(torch, config.emb_config.emb_precision),
                }
            },
        )
        self.vectordb = Chroma(
            embedding_function=self.embedding,
            persist_directory=str(config.vdb_config.vdb_path)
        )

        if not config.reranker_config.no_rerank:
            self.reranker = SentenceTransformer(
                str(config.reranker_config.reranker_path),
                device=config.reranker_config.reranker_device,
                model_kwargs={
                    "torch_dtype": getattr(
                        torch, config.reranker_config.reranker_precision
                    )
                }
            )

        self.last_docs_to_pull = None

        self.config = config
        self.reranker_config = config.reranker_config
        self.llm_config = config.llm_config

    def rerank(self, query, docs, retrieval_threshold, n_retrieved_docs):
        # Prepare a prompt given an instruction
        prompt = f'<instruct>{self.reranker_config.reranker_guide}\n<query>'

        # Compute the query and document embeddings
        query_embeddings = self.reranker.encode(query, prompt=prompt)
        document_embeddings = self.reranker.encode(
            [doc[0].page_content for doc in docs]  # Get just text, no retrieval score
        )

        # Compute the cosine similarity between the query and document embeddings
        similarities = self.reranker.similarity(
            query_embeddings, document_embeddings
        )[0]
        best_scores, best_indices = torch.topk(similarities, n_retrieved_docs)
        docs = [
            (
                docs[best_indices[i]][0],
                (docs[best_indices[i]][1], best_scores[i].cpu().item())
            )
            for i in range(len(best_indices)) if best_scores[i] > retrieval_threshold
        ]
        return docs

    def get_retrived_docs(
        self,
        query,
        retrieval_threshold,
        n_retrieved_docs,
        coef_rerank_retrieve_docs,
        rerank
    ):
        if rerank:
            docs = self.vectordb.similarity_search_with_relevance_scores(
                query, k=int(n_retrieved_docs * coef_rerank_retrieve_docs)
            )

            docs = self.rerank(query, docs, retrieval_threshold, n_retrieved_docs)

        else:
            docs = self.vectordb.similarity_search_with_relevance_scores(
                query, k=n_retrieved_docs
            )
            docs = [
                (doc, (score, "N/A"))
                for doc, score in docs if score > retrieval_threshold
            ]
            docs = sorted(docs, key=lambda x: x[1][0], reverse=True)

        self.last_retrieved_docs = [
            {
                "metadata": doc[0].metadata,
                "content": doc[0].page_content,
                "retrieval score": doc[1][0],
                "rerank score": doc[1][1],
            } for doc in docs
        ]  # Save the last retrieved docs for dpo training

        self.last_docs_to_pull = docs
        return docs

    def remove_source(self, text):
        return re.sub(r'\n\nsources with their scores:\n.*', '', text, flags=re.DOTALL)

    def get_llm_input(self, message, docs, history):
        retrieved_infos = " ".join([doc[0].page_content for doc in docs])

        header = self.llm_config.llm_header + "\n"

        retrieved_str = f"<|retrieved_info|>: {retrieved_infos}\n"

        if len(history) > 0:
            history_str = "\n".join(
                f"<|user|>: {interaction[0]}\n"
                + f"<|assistant|>:{self.remove_source(interaction[1])}\n"
                for interaction in history
            )
        else:
            history_str = ""

        self.last_history = history_str
        last_chat = f"<|user|>: {message}\n<|assistant|>:"

        text_input = header + retrieved_str + history_str + last_chat
        return text_input

    def save_chat(self, liked: bool | None = None):
        if self.name_save_file is not None:
            like_element = {
                "query": self.last_query,
                "generation": self.last_generation,
                "retrieved_docs": self.last_retrieved_docs,
                "history": self.last_history,
                "like": liked,
                "config": self.config.export_config()
            }

            # Save the like data in a json, the name is the current timestamp
            with open(
                self.config.saved_data_path / f"{self.name_save_file}.json", "w"
            ) as f:
                json.dump(like_element, f, ensure_ascii=False)

    async def chat(
        self,
        message,
        history,
        temperature,
        max_tokens,
        retrieval_threshold,
        n_retrieved_docs,
        coef_rerank_retrieve_docs,
        rerank=False
    ):
        self.name_save_file = None  # here to prevent bug

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=self.llm_config.stop_tokens
        )

        query = message + " " + " ".join(
            [interaction[0] for interaction in history]
            if len(history) > 0 else ""
        )
        self.last_query = query  # Save the last query for dpo training
        retrieved_docs = self.get_retrived_docs(
            query,
            retrieval_threshold,
            n_retrieved_docs,
            coef_rerank_retrieve_docs,
            rerank
        )

        if len(retrieved_docs) == 0:
            no_doc_message = "No relevant documents found. Please retry with another "
            no_doc_message += "query or call the assistance."
            self.last_generation = no_doc_message
            yield no_doc_message

        else:
            # get a unique request id per chat
            request_id = str(hash(query))
            outputs = self.llm.generate(
                self.get_llm_input(message, retrieved_docs, history),
                sampling_params,
                request_id
            )

            async for request_output in outputs:
                yield request_output.outputs[0].text

            self.last_generation = request_output.outputs[0].text

            chat_out = request_output.outputs[0].text

            yield chat_out

        print(f"Query: {self.last_query}\nResponse: {self.last_generation}")
        self.name_save_file = datetime.now()
        self.save_chat()
