import time

import gradio as gr

from .config import GlobalConfig
from .engine import PieChat

from gradio_rag_sources import RagSourcesTable, _RagSource as RagSource

WELCOME_MESSAGE = """<strong>
PieChat - Votre assistant à l'utilisation de Jean Zay
</strong><br>
Vous pouvez lui faire confiance à 100%, il ne se trompe jamais."""


def await_rag_sources(piechat: PieChat):

    def wrapped() -> list[RagSource]:
        while True:
            sources = piechat.last_docs_to_pull
            if sources is None:
                time.sleep(0.25)
            else:
                piechat.last_docs_to_pull = None
                break
        docs: list[RagSource] = [
            RagSource(
                url=doc[0].metadata["source"],
                retrievalScore=doc[1][0],
                rerankScore=doc[1][1],
            )
            for doc in sources
        ]
        return docs

    return wrapped


def launch_gradio(piechat: PieChat, config: GlobalConfig):

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                additional_inputs = [
                    gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=config.llm_config.temperature,
                        label="Temperature"
                    ),
                    gr.Slider(
                        minimum=1,
                        maximum=4096,
                        value=config.llm_config.max_tokens,
                        label="Max Tokens"
                    ),
                    gr.Slider(
                        minimum=-1.0,
                        maximum=1.0,
                        value=config.retrieval_threshold,
                        label="Similarity Retrieval Threshold"
                    ),
                    gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=config.n_retrieved_docs,
                        label="Maximum number of Retrieved Docs to give to the llm"
                    ),
                    gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=config.coef_rerank_retrieve_docs,
                        label="This coef x max docs for llm is the nb of Docs to rerank"
                    )
                ]

                if not config.reranker_config.no_rerank:
                    # Add the activation reranker button
                    additional_inputs.append(
                        gr.Checkbox(
                            value=True,
                            label="Activate reranker",
                            scale=1
                        )
                    )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    placeholder=WELCOME_MESSAGE
                )

                def save_like_data(data: gr.LikeData):
                    piechat.save_chat(data.liked)

                chatbot.like(save_like_data, None, None)
                chatInterface = gr.ChatInterface(
                    piechat.chat,
                    chatbot=chatbot,
                    additional_inputs=additional_inputs
                )
                (
                    retry_btn,
                    undo_btn,
                    clear_btn,
                    submit_btn,
                    stop_btn,
                ) = chatInterface.buttons

                ragSourcesGradioComponent = RagSourcesTable(value=[]) #gr.Textbox(label="Rag sources")

                awaitEvent = gr.on(
                    triggers=[submit_btn.click, retry_btn.click, chatInterface.textbox.submit],
                    fn=await_rag_sources(piechat),
                    outputs=[ragSourcesGradioComponent],
                )
                gr.on(
                    triggers=[clear_btn.click, undo_btn.click],
                    fn=lambda: [],
                    outputs=[ragSourcesGradioComponent],
                )
                gr.on(
                    triggers=[stop_btn.click],
                    cancels=[awaitEvent],
                )

    demo.launch(share=True)
