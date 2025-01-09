import sys
import os  # We need this for fsync
import gradio as gr

# This still helps with Python-level buffering
sys.stdout.reconfigure(line_buffering=True)

# Get the file descriptor for stdout
stdout_fd = sys.stdout.fileno()

def force_sync():
    """
    Force the stdout buffer to be written to disk and synchronized with the AFS server.
    This ensures the output is immediately visible from other machines.
    """
    # First flush Python's buffers
    sys.stdout.flush()
    # Then tell the OS to sync the file descriptor
    os.fsync(stdout_fd)

def greet(name, intensity):
    force_sync()
    result = "Hello, " + str(name) + "!"*int(intensity)
    return result

demo = gr.Interface(
    fn=greet,
    inputs=["text","slider"],
    outputs=["text"],
)

# Launch Gradio and get the URLs
_, local_url, public_url = demo.launch(share=True); force_sync()

# Force sync after writing the URLs
force_sync()
