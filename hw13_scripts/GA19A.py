# GA19A.py CS6078 cheng 2026
# Chapter 9 of Hands-on GenAI
# Usage: python GA19A.py

from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
print(minds)

example = minds[0]
print(example)

id2label = minds.features["intent_class"].int2str
print(id2label(example["intent_class"]))

columns_to_remove = ["lang_id", "english_transcription"]
minds = minds.remove_columns(columns_to_remove)
print(minds)

import gradio as gr

def generate_audio():
    example = minds.shuffle()[0]
    audio = example["audio"]
    return (
        audio["sampling_rate"],
        audio["array"],
    ), id2label(example["intent_class"])


with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)