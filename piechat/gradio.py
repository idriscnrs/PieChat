import time

import gradio as gr

from .config import GlobalConfig
from .engine import PieChat

WELCOME_MESSAGE = """<strong>
PieChat - Votre assistant à l'utilisation de Jean Zay
</strong><br>
Vous pouvez lui faire confiance à 100%, il ne se trompe jamais."""


def await_rag_sources(piechat: PieChat):

    def wrapped():
        while True:
            sources = piechat.last_docs_to_pull
            if sources is None:
                time.sleep(0.25)
            else:
                piechat.last_docs_to_pull = None
                break
        docs = [
            f"{doc[0].metadata['source']} retrieval_score={doc[1][0]}"
            + f" rerank_score={doc[1][1]}"
            for doc in sources
        ]
        return "\n".join(docs)

    return wrapped


def launch_gradio(piechat: PieChat, config: GlobalConfig):

    submitButton = gr.Button(
        "Submit",
        variant="primary",
        scale=1,
        min_width=150,
        render=False,
    )
    inputRequestGradioComponent = gr.Textbox(
        container=False,
        show_label=False,
        label="Message",
        placeholder="Type a message...",
        scale=7,
        autofocus=True,
        render=False,
    )

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
                gr.ChatInterface(
                    piechat.chat,
                    chatbot=chatbot,
                    textbox=inputRequestGradioComponent,
                    submit_btn=submitButton,
                    additional_inputs=additional_inputs
                )

                ragSourcesGradioComponent = gr.Textbox(label="Rag sources")

                gr.on(
                    triggers=[submitButton.click, inputRequestGradioComponent.submit],
                    fn=await_rag_sources(piechat),
                    outputs=[ragSourcesGradioComponent],
                )

    demo.launch(share=True)
