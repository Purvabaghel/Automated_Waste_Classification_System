import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gradio as gr
from PIL import Image
from warnings import filterwarnings

filterwarnings('ignore')

# Paths to the trained model
model_path = 'model/finetuned_model.h5'

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices = np.load('model/class_indices.npy', allow_pickle=True).item()
idx_to_class = {v: k for k, v in class_indices.items()}

# Recycling tips for each class
RECYCLING_TIPS = {
    "paper": [
        "Remove any plastic windows from envelopes before recycling",
        "Flatten cardboard boxes to save space in recycling bins",
        "Keep paper dry and clean from food residues"
    ],
    "cardboard": [
        "Break down boxes completely before recycling",
        "Remove tape and plastic labels when possible",
        "Pizza boxes with grease stains should be composted"
    ],
    "biological": [
        "Compost fruit and vegetable scraps in a proper compost bin",
        "Eggshells should be crushed before adding to compost",
        "Avoid composting meat, dairy, or oily foods"
    ],
    "metal": [
        "Rinse cans to remove food residue before recycling",
        "Separate aluminum and steel cans if required in your area",
        "Remove plastic lids from metal containers"
    ],
    "plastic": [
        "Check the recycling number (1-7) on plastic items",
        "Rinse plastic containers to remove food residue",
        "Plastic bags should be recycled separately at grocery stores"
    ],
    "green-glass": [
        "Rinse bottles before recycling",
        "Remove metal caps and lids",
        "Don't mix with other colored glass"
    ],
    "brown-glass": [
        "Rinse thoroughly before recycling",
        "Labels can usually stay on",
        "Brown glass has high recycling value"
    ],
    "white-glass": [
        "Rinse clean but no need to remove labels",
        "Keep separate from colored glass",
        "Clear glass has the highest recycling potential"
    ],
    "clothes": [
        "Donate wearable clothing to charities",
        "Repurpose old clothes as cleaning rags",
        "Look for textile recycling programs"
    ],
    "shoes": [
        "Donate wearable shoes to shelters",
        "Some brands offer shoe recycling programs",
        "Separate components before recycling"
    ],
    "batteries": [
        "Never put batteries in regular trash",
        "Find battery recycling drop-off locations",
        "Tape terminals of lithium batteries before disposal"
    ],
    "trash": [
        "Consider if items can be repaired before discarding",
        "Reduce single-use items in the future",
        "Check local guidelines for proper disposal"
    ]
}

def classify_image(image):
    # Convert Gradio's numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize the image to match model's expected input
    image = image.resize((224, 224))
    
    # Prepare the image
    img_array = img_to_array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict the class of the image
    predictions = model.predict(img_array)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_class = idx_to_class[predicted_class_idx]
    confidence = predictions[predicted_class_idx]
    
    # Create a dictionary of class probabilities
    class_probs = {idx_to_class[idx]: float(prob) for idx, prob in enumerate(predictions)}
    
    # Get recycling tips for the predicted class and format as markdown
    tips = RECYCLING_TIPS.get(predicted_class, ["No specific recycling tips available for this item."])
    tips_markdown = "**Recycling Tips:**\n\n" + "\n".join(f"- {tip}" for tip in tips)
    
    return predicted_class, confidence, class_probs, tips_markdown

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Classification with Fine-Tuned Model")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil")
            submit_btn = gr.Button("Classify")
        
        with gr.Column():
            label_output = gr.Textbox(label="Predicted Class")
            confidence_output = gr.Number(label="Confidence Score")
            probs_output = gr.Label(label="Class Probabilities")
    
    # Add tips section (initially hidden)
    with gr.Column(visible=False) as tips_column:
        tips_output = gr.Markdown()  # Using Markdown to keep text black
    
    submit_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[label_output, confidence_output, probs_output, tips_output]
    ).then(
        lambda: gr.Column(visible=True),  # Show tips section after classification
        outputs=tips_column
    )
    
    gr.Examples(
        examples=["C:/Users/HP/Documents/saket/brown-glass/brown-glass4.jpg"],
        inputs=image_input,
        label="Example Images"
    )

demo.launch()