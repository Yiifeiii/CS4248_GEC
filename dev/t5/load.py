import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Run on GPU if available else run on CPU   
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the Hugging Face pretrained model 
name = "vennify/t5-base-grammar-correction"   


tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()