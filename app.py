import gradio as gr
import numpy as np

from transformers import pipeline

# caption = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
caption = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


def run_caption(img):
  res = caption(img, max_new_tokens=128)
  return res[0]["generated_text"]



with gr.Blocks() as demo:
  gr.Interface(
    run_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
  )


if __name__ == "__main__":
   demo.launch()
