## Update Notes
### 2024/10/29
1. we use the stateless mode, so two models will be generated, one for prefill and the other for generate.
2. Due to the Mamba structure, only input lengths of 6 are supported.
3. Since transformers use MambaCache as model input, but ov does not support it, we refactored the modeling_mamba.py file

## Running Guide
### Installation


```bash
git clone https://github.com/zhaohb/mamba_ov.git
pip install openvino_dev 
pip install transformers==4.39.0

Additional Operations
1. download mamba-2.8b-hf model
2. Copy all files under mamba-2.8b to the mamba-2.8b-hf directory
```
### Convert mamba-2.8b model to OpenVINO™ IR(Intermediate Representation) and testing:
```shell
cd mamba-ov
python3 test_mamba.py -m state-spaces/mamba-2.8b-hf -ov test_ov
```
The model: [Model link](https://hf-mirror.com/state-spaces/mamba-2.8b-hf)
### Parsing test_mamba.py's arguments :
```shell
usage: Export mamba-2.8b Model to IR [-h] [-m MODEL_ID] -ov OV_IR_DIR [-d DEVICE] [-p PROMPT] [-max MAX_NEW_TOKENS] [-llm_int4_com]

options:
  -h, --help            show this help message and exit
  -m MODEL_ID, --model_id MODEL_ID
                        model_id or directory for loading
  -ov OV_IR_DIR, --ov_ir_dir OV_IR_DIR
                        output directory for saving model
  -d DEVICE, --device DEVICE
                        inference device
  -p PROMPT, --prompt PROMPT
                        prompt
  -max MAX_NEW_TOKENS, --max_new_tokens MAX_NEW_TOKENS
                        max_new_tokens
  -llm_int4_com, --llm_int4_compress
                        llm int4 weights compress
```

