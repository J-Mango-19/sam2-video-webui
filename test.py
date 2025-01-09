import sys
import gradio as gr

sys.stdout.reconfigure(line_buffering=True)

def greet(name, intensity):
    return "Hello, " + str(name) + "!"*int(intensity)

demo = gr.Interface(
        fn=greet,
        inputs=["text","slider"],
        outputs=["text"],
        )

_, local_url, public_url = demo.launch(share=True)
