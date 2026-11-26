import gradio as gr
from PIL import Image

IMAGE_READ_ERRORS = (OSError, ValueError, Image.DecompressionBombError)

class LeafImage(gr.Image):
    """An image input that shows a clear message when a file can't be read.
    """
    def get_block_name(self):
        return "image"  # draw it with Gradio's normal image box in the browser

    def preprocess(self, payload):
        try:
            return super().preprocess(payload)
        except IMAGE_READ_ERRORS as err:
            raise gr.Error("Couldn't read this file. Upload a JPG or PNG photo of a leaf.") from err