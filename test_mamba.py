import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import argparse
import openvino as ov
from pathlib import Path

import numpy as np
from ov_mamba import OVMambaForCausalLM, MambaModel

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Export InternVL2 Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=False, help="model_id or directory for loading")
    parser.add_argument("-ov", "--ov_ir_dir", required=True, help="output directory for saving model")
    parser.add_argument('-d', '--device', default='CPU', help='inference device')
    parser.add_argument('-p', '--prompt', default="Hey how are you doing?", help='prompt')
    parser.add_argument('-max', '--max_new_tokens', default=10, help='max_new_tokens')
    parser.add_argument('-llm_int4_com', '--llm_int4_compress', action="store_true", help='llm int4 weights compress')

    args = parser.parse_args()
    model_id = args.model_id
    ov_model_path = args.ov_ir_dir
    device = args.device
    max_new_tokens = args.max_new_tokens
    question = args.prompt
    llm_int4_compress = args.llm_int4_compress

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    if not Path(ov_model_path).exists():
        mamba_ov = MambaModel(model=model, tokenizer=tokenizer, ov_model_path=ov_model_path, device=device, int4_compress=llm_int4_compress)
        mamba_ov.convert_ov()
        mamba_ov.convert_past_ov()
        del mamba_ov.model
        del mamba_ov.tokenizer
        del mamba_ov
    elif Path(ov_model_path).exists() and llm_int4_compress is True and not Path(f"{ov_model_path}/ov_mamba_int4.xml").exists():
        mamba_ov = MambaModel(model=model, tokenizer=tokenizer, ov_model_path=ov_model_path, device=device, int4_compress=llm_int4_compress)
        mamba_ov.convert_ov()
        mamba_ov.convert_past_ov()
        del mamba_ov.model
        del mamba_ov.tokenizer
        del mamba_ov
    del model, tokenizer

    core = ov.Core()
    ov_mamba_model = OVMambaForCausalLM(ov_model_path=ov_model_path, core=core)
    input_ids = ov_mamba_model.tokenizer(question, return_tensors="pt")["input_ids"]
    out = ov_mamba_model.generate(input_ids, max_new_tokens=max_new_tokens)
    print(out)
    print(ov_mamba_model.tokenizer.batch_decode(out))

