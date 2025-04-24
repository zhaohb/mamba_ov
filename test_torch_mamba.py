# from transformers import MambaConfig, OvMambaForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import openvino as ov
from pathlib import Path
import nncf

from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("mamba-2.8b-hf")
model = MambaForCausalLM.from_pretrained("mamba-2.8b-hf")
input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=256)
print(out)
print(tokenizer.batch_decode(out))
