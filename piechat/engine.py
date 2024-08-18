import re
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from .config import LLMConfig


class PieChat:
    def __init__(
        self,
        embedding_path: Path,
        vdb_path: Path,
        llm_config: LLMConfig,
    ):
        llm_args = AsyncEngineArgs(
            model=str(llm_config.llm_path),
            max_model_len=llm_config.max_model_length,
            dtype=llm_config.dtype,
            gpu_memory_utilization=llm_config.gpu_memory_utilization
        )
        self.llm = AsyncLLMEngine.from_engine_args(llm_args)
        self.sampling_params = SamplingParams(
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            stop=llm_config.stop_tokens
        )

        self.embedding = HuggingFaceEmbeddings(
            model_name=str(embedding_path),  # Does not accept Path
            model_kwargs={"device": "cuda", "trust_remote_code": True},
        )
        self.vectordb = Chroma(
            embedding_function=self.embedding,
            persist_directory=str(vdb_path)
        )

    def get_retrived_docs(self, message):
        docs = self.vectordb.similarity_search_with_relevance_scores(message, k=6)
        return docs

    def remove_source(self, text):
        return re.sub(r'\n\nsources:\n.*', '', text, flags=re.DOTALL)

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

        last_chat = f"<|user|>: {message}\n<|assistant|>:"

        text_input = header + retrieved_str + history_str + last_chat
        print(text_input)
        return text_input

    def get_sources(self, docs):
        # Extract source and score, and sort by score
        sorted_sources = sorted(
            [f"{doc[0].metadata['source']} {doc[1]}" for doc in docs],
            key=lambda x: float(x.split()[-1]),  # The score is the last element
            reverse=True  # Sort in descending order
        )
        sources = "\n".join(sorted_sources)
        return "sources with their scores:\n" + sources

    async def chat(self, message, history):

        query = message + " " + " ".join(
            [interaction[0] for interaction in history]
            if len(history) > 0 else ""
        )
        retrieved_docs = self.get_retrived_docs(query)

        # get a unique request id per chat
        request_id = str(hash(query))
        outputs = self.llm.generate(
            self.get_llm_input(message, retrieved_docs, history),
            self.sampling_params,
            request_id
        )

        async for request_output in outputs:
            yield request_output.outputs[0].text

        sources = self.get_sources(retrieved_docs)

        chat_out = request_output.outputs[0].text + "\n"*2 + sources

        yield chat_out
