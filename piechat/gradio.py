import json
from datetime import datetime

import gradio as gr

from .config import GlobalConfig
from .engine import PieChat

WELCOME_MESSAGE = """<strong>
PieChat - Votre assistant à l'utilisation de Jean Zay
</strong><br>
Vous pouvez lui faire confiance à 100%, il ne se trompe jamais."""


def launch_gradio(piechat: PieChat, config: GlobalConfig):
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
                with gr.Row():
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
                        additional_inputs=[temperature, max_tokens, retrieval_threshold]
                    )
    
                with gr.Row():
                    ragSourceGradioComponent = gr.Textbox(label="Rag sources")
                    chatbot.change(piechat.get_sources, outputs=[ragSourceGradioComponent])
                

    demo.launch(share=True)
