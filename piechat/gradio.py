import json
import time
from datetime import datetime

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
            try:
                sources = piechat.get_sources()
            except AttributeError:
                time.sleep(0.25)
            else:
                return "\n".join(sources.split("\n")[1:])

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
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=config.llm_config.temperature,
                    label="Temperature"
                )
                max_tokens = gr.Slider(
                    minimum=1,
                    maximum=4096,
                    value=config.llm_config.max_tokens,
                    label="Max Tokens"
                )
                retrieval_threshold = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    value=config.retrieval_threshold,
                    label="Similarity Retrieval Threshold"
                )
    
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    placeholder=WELCOME_MESSAGE
                )

                def save_like_data(data: gr.LikeData):
                    like_element = {
                        "query": piechat.last_query,
                        "generation": piechat.last_generation,
                        "retrieved_docs": piechat.last_retrieved_docs,
                        "history": piechat.last_history,
                        "like": data.liked,
                        "config": config.export_config()
                    }

                    # Save the like data in a json, the name is the current timestamp
                    with open(
                        config.like_data_path / f"{datetime.now()}.json", "w"
                    ) as f:
                        json.dump(like_element, f, ensure_ascii=False)

                chatbot.like(save_like_data, None, None)
                gr.ChatInterface(
                    piechat.chat,
                    chatbot=chatbot,
                    textbox=inputRequestGradioComponent,
                    submit_btn=submitButton,
                    additional_inputs=[temperature, max_tokens, retrieval_threshold]
                )
    
                ragSourcesGradioComponent = gr.Textbox(label="Rag sources")

                gr.on(
                    triggers=[submitButton.click, inputRequestGradioComponent.submit],
                    fn=await_rag_sources(piechat),
                    outputs=[ragSourcesGradioComponent],
                )

    demo.launch(share=True)
