import gradio as gr
from PIL import Image

from gradio.events import Dependency

class LeafImage(gr.Image):
    """An image input that shows a clear message when a file can't be read.
    """
    def get_block_name(self):
        return "image"  # draw it with Gradio's normal image box in the browser

    def preprocess(self, payload):
        try:
            return super().preprocess(payload)
        except (OSError, Image.DecompressionBombError) as err:
            raise gr.Error("Couldn't read this file. Upload a JPG or PNG photo of a leaf.") from err
    from typing import Callable, Literal, Sequence, Any, TYPE_CHECKING
    from gradio.blocks import Block
    if TYPE_CHECKING:
        from gradio.components import Timer
        from gradio.components.base import Component