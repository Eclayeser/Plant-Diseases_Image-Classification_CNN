import gradio as gr
from torchvision import transforms
import pandas as pd
import torch

def get_diseases():
    return [
        "Tomato - Healthy",
        "Tomato - Early Blight",
        "Tomato - Late Blight",
        "Tomato - Leaf Mold",
        "Tomato - Septoria Leaf Spot"
    ]
def get_probabilities():
    return [0.75, 0.12, 0.07, 0.04, 0.02]

# TODO: Connect this function to the trained CNN model once it is ready.
# For now, the Gradio interface uses dummy predictions.
def get_cnn_predictions(processed_image, model, class_names):
    model.eval()
    with torch.no_grad():
        output = model(processed_image.unsqueeze(0))

    probs = torch.softmax(output, dim=1)[0]
    top5 = torch.topk(probs, 5)

    diseases = []
    for i in top5.indices:
        diseases.append(class_names[int(i)])

    return pd.DataFrame({
        "Disease": diseases,
        "Probability": top5.values.tolist()
    })

def get_predictions(processed_image):
    #  TODO: switch to get_cnn_predictions when the model is ready.
    diseases = get_diseases()
    probabilities = get_probabilities()

    return pd.DataFrame({
        "Disease": diseases,
        "Probability": probabilities
    })




def predict(image):
    if image is None:
        return None
    
    # Preprocess the image to match the CNN input requirements
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])
    image = image.convert("RGB")
    processed_image = transform(image)

    return get_predictions(processed_image)

# Create the Gradio interface for image upload and prediction display
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
    description="UI prototype: Predictions are simulated using dummy values. The trained CNN model is not connected yet.",
    api_name="predict"
)

if __name__ == "__main__":
    demo.launch()