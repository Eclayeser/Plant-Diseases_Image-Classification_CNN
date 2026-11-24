import gradio as gr
from PIL import Image


class LeafImage(gr.Image):
    """An image input that shows a clear message when a file can't be read.
    """

    def preprocess(self, payload):
        try:
            return super().preprocess(payload)
        except (OSError, Image.DecompressionBombError) as err:
            raise gr.Error("Couldn't read this file. Upload a JPG or PNG photo of a leaf.") from err