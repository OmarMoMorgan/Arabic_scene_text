import matplotlib.pyplot as plt
from PIL import Image
import torch 


def generate_text_with_caption(image_path,processor,device,model):

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    

    outputs = model.generate(pixel_values, num_beams=4, max_length=512, early_stopping=True)
    predicted_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(predicted_text)
    plt.show()
    
    return predicted_text