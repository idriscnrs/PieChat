import re
from pathlib import Path

import idr_torch
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .config import LLMConfig, EmbeddingConfig, RerankerConfig


class PieChat:
    def __init__(
        self,
        vdb_path: Path,
        llm_config: LLMConfig,
        emb_config: EmbeddingConfig,
        reranker_config: RerankerConfig,
    ):
        llm_args = AsyncEngineArgs(
            model=str(llm_config.llm_path),
            max_model_len=llm_config.max_model_length,
            dtype=llm_config.dtype,
            gpu_memory_utilization=llm_config.gpu_memory_utilization,
            tensor_parallel_size=llm_config.tensor_parallel_size
        )
        self.llm = AsyncLLMEngine.from_engine_args(llm_args)
        self.llm_config = llm_config

        self.embedding = HuggingFaceEmbeddings(
            model_name=str(emb_config.embedding_path),  # Does not accept Path
            model_kwargs={
                "device": f"cuda:{emb_config.emb_device_id}",
                "trust_remote_code": not emb_config.no_trust_remote_code,
                "model_kwargs": {
                    "attn_implementation": emb_config.attn_implementation,
                    "torch_dtype": torch.float16
                } 
            },
        )
        self.vectordb = Chroma(
            embedding_function=self.embedding,
            persist_directory=str(vdb_path)
        )

        self.reranker = SentenceTransformer(
            str(reranker_config.reranker_path),
            device=f"cuda:{reranker_config.reranker_device_id}",
            model_kwargs={"torch_dtype": torch.float16}
        )
        self.last_docs_to_pull = None

    def rerank(self, query, docs, retrieval_threshold):
        # Prepare a prompt given an instruction
        instruction = 'Given an instruction or a question, retrieve relevant passages that help to answer the query. The relevant passage should be the same language than the instruction or question.'
        prompt = f'<instruct>{instruction}\n<query>'

        # Compute the query and document embeddings
        query_embeddings = self.reranker.encode(query, prompt=prompt)
        print("#"*10, 0, docs)
        document_embeddings = self.reranker.encode([doc[0].page_content for doc in docs])  # Get just text, no retrieval score

        # Compute the cosine similarity between the query and document embeddings
        similarities = self.reranker.similarity(query_embeddings, document_embeddings)[0]
        best_scores, best_indices = torch.topk(similarities, 8)
        docs = [
            (docs[best_indices[i]][0], (docs[best_indices[i]][1], best_scores[i]))
            for i in range(len(best_indices)) if best_scores[i] > retrieval_threshold
        ]
        return docs

    def get_retrived_docs(self, query, retrieval_threshold):
        docs = self.vectordb.similarity_search_with_relevance_scores(query, k=32)

        docs = self.rerank(query, docs, retrieval_threshold)
        self.last_retrieved_docs = [
            {
                "metadata": doc[0].metadata,
                "content": doc[0].page_content,
                "retrieval score": doc[1][0],
                "rerank score": doc[1][1],
            } for doc in docs
        ]  # Save the last retrieved docs for dpo training

        # docs = [(doc, score) for doc, score in docs if score > retrieval_threshold]
        self.last_docs = docs
        self.last_docs_to_pull = docs
        return docs

    def remove_source(self, text):
        return re.sub(r'\n\nsources with their scores:\n.*', '', text, flags=re.DOTALL)

    def get_llm_input(self, message, docs, history):
        retrieved_infos = " ".join([doc[0].page_content for doc in docs])

        header = (
            "<|begin_of_text|><|start_header_id|>You are a chatbot that help Jean Zay "
            + "users. Jean Zay is the supercomputer hosted by IDRIS-CNRS. The users "
            + "speaks French or English and you should answer with the same language."
            + "You should answer the question using the <|retrieved_info|>."
            + "<|end_header_id|>\n"
        )

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
        print(text_input)
        return text_input

    def get_sources(self):
        sorted_sources = [
            f"{doc[0].metadata['source']} retrieval_score={doc[1][0]} rerank_score{doc[1][1]}"
            for doc in self.last_docs
        ]
        # Extract source and score, and sort by score
        # sorted_sources = sorted(
        #     [f"{doc[0].metadata['source']} {doc[1]}" for doc in docs],
        #     key=lambda x: float(x.split()[-1]),  # The score is the last element
        #     reverse=True  # Sort in descending order
        # )

        sources = "\n".join(sorted_sources)
        return "sources with their scores:\n" + sources

    async def chat(
        self, message, history, temperature, max_tokens, retrieval_threshold
    ):
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
        retrieved_docs = self.get_retrived_docs(query, retrieval_threshold)

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

            sources = self.get_sources()
            chat_out = request_output.outputs[0].text  # + "\n"*2 + sources

            yield chat_out
