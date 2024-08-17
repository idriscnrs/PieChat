import gradio as gr
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from pathlib import Path
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re

DSDIR = Path(os.environ['DSDIR'])
HF_MODELS_PATH = DSDIR / "HuggingFace_Models"

EMBEDDING_PATH = HF_MODELS_PATH / "sentence-transformers/sentence-t5-xxl"
VDB_PATH = Path("./SpeLLM/chroma_vdb_t5")
MODEL_PATH = HF_MODELS_PATH / "meta-llama/Meta-Llama-3-8B-Instruct"


class PieChat:
    def __init__(self):
        llm_args = AsyncEngineArgs(
            model=str(MODEL_PATH),
            dtype="bfloat16",
            gpu_memory_utilization=0.50
        )
        self.llm = AsyncLLMEngine.from_engine_args(llm_args)
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=1000,
            stop=["<|eot_id|>", "<|user|>", "<|assistant|>"]
        )

        self.embedding = HuggingFaceEmbeddings(
            model_name=str(EMBEDDING_PATH),  # Does not accept Path
            model_kwargs={"device": "cuda"},
        )
        self.vectordb = Chroma(
            embedding_function=self.embedding,
            persist_directory=str(VDB_PATH)
        )

    def get_retrived_docs(self, message):
        docs = self.vectordb.similarity_search(message, k=8)
        return docs

    def remove_source(self, text):
        return re.sub(r'\n\nsources:\n.*', '', text, flags=re.DOTALL)

    def get_llm_input(self, message, docs, history):
        retrieved_infos = " ".join([doc.page_content for doc in docs])

        header = (
            "<|start_header_id|>You are a chatbot that help Jean Zay users. "
            + "Jean Zay is the supercomputer hosted by IDRIS-CNRS. The users speaks "
            + "French or English and you should answer with the same language."
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
        sources = "\n".join(
            set([doc.metadata["source"] for doc in docs])
        )
        return "sources:\n" + sources

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


if __name__ == "__main__":
    pie_chat = PieChat()
    gr.ChatInterface(pie_chat.chat).launch(share=True)
