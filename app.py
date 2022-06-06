import os
import torch
import gradio as gr

from PIL import Image
from torchvision import transforms


"""
Built following:
https://huggingface.co/spaces/pytorch/ResNet/tree/main
https://www.gradio.app/image_classification_in_pytorch/
"""

# Get classes list
os.system("wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

# Load PyTorch model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Download an example image from the pytorch website
torch.hub.download_url_to_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")

# Inference!
def inference(input_image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = {}
    for i in range(top5_prob.size(0)):
        result[categories[top5_catid[i]]] = top5_prob[i].item()
    return result

# Define ins outs placeholders
inputs = gr.inputs.Image(type='pil')
outputs = gr.outputs.Label(type="confidences",num_top_classes=5)

# Define style
title = "Image Recognition Demo"
description = "This is a prototype application which demonstrates how artifical intelligence based systems can recognize what object(s) is present in an image. This fundamental task in computer vision known as `Image Classification` has applications stretching from autonomous vehicles to medical imaging. To use it, simply upload your image, or click one of the examples images to load them, which I took at <a href='https://espacepourlavie.ca/en/biodome' target='_blank'>Montréal Biodôme</a>! Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1512.03385' target='_blank'>Deep Residual Learning for Image Recognition</a> | <a href='https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py' target='_blank'>Github Repo</a></p>"

# Run inference
gr.Interface(inference, 
            inputs, 
            outputs, 
            examples=["example1.jpg", "example2.jpg"], 
            title=title, 
            description=description, 
            article=article,
            analytics_enabled=False).launch()

