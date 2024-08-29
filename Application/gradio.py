import gradio as gr
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load the fine-tuned model and tokenizer
model_name = "./fine-tuned-biomistral"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model.eval()

# Define the chatbot function
def chatbot(input_text):
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Generate the model output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the logits
    logits = outputs.logits

    # Decode the predicted tokens to get the output text
    predicted_ids = torch.argmax(logits, dim=-1)
    output_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    return output_text

# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot, 
    inputs="text", 
    outputs="text", 
    title="Medical Diagnosis Chatbot",
    description="Chat with the fine-tuned Bio_ClinicalBERT model for medical diagnosis.",
)

# Launch the Gradio app
iface.launch()
