import gradio as gr
from torchvision import transforms

def predict(image):
    if image is None:
        return None
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])
    image = image.convert("RGB")
    processed_image = transform(image)

    return None

demo = gr.Interface(
    fn=predict,
    
    inputs=gr.Image(type="pil", label="Upload a leaf image"),
    outputs=gr.BarPlot(
    x="Disease",
    y="Probability",
    title="Top 5 Predictions",
    x_label_angle=45
    ),
    
    title="Plant Disease Classification",
    description="Upload a plant leaf image to identify its disease.",
    api_name="predict"
)

demo.launch()