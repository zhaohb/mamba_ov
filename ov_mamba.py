# from transformers import MambaConfig, OvMambaForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import openvino as ov
from pathlib import Path
import nncf
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.utils import ModelOutput
from typing import Optional, Tuple, List, Union, Dict, Any
import time
import types
import numpy as np
from dataclasses import dataclass
from openvino.runtime import opset13

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])

def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    main_input_name = "input_ids" if model_has_input_output_name(ov_model, "input_ids") else "inputs_embeds"
    input_ids = ov_model.input(main_input_name)
    batch = opset13.gather(opset13.shape_of(input_ids, output_type="i64"), opset13.constant([0]), opset13.constant(0))
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            # breakpoint()
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()

def patch_stateful_ssm(ov_model):
    cache_input_names = [key_name for key in ov_model.inputs for key_name in key.get_names() if "past" in key_name]
    cache_output_names = [
        key_name for key in ov_model.outputs for key_name in key.get_names() if "present" in key_name
    ]

    # breakpoint()
    # print(cache_output_names)
    # print(ov_model.outputs)
    if not cache_input_names or not cache_output_names:
        return

    batch_dim = 0

    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}
    for cache_name_pair in zip(cache_input_names, cache_output_names):
        input_output_map[cache_name_pair[0]] = cache_name_pair[1]

    # print(input_output_map)

    apply_make_stateful_transformation(ov_model, input_output_map)
    build_state_initializer(ov_model, batch_dim)



class MambaModel():
    def __init__(
        self,
        model=None,
        tokenizer= None,
        device='CPU',
        ov_model_path=None,
        int4_compress=False,
        fp16=False,
    ):
        self.name = "Mamba Model"
        self.model = model
        self.tokenizer = tokenizer
        self.device=device
        self.ov_model_path = ov_model_path
        self.int4_compress = int4_compress
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model

    def get_input_names(self):
        return [ 'inputs_embeds',]
    
    def get_past_input_names(self):
        inputs = [ 'input_ids', 'cache_position']
        for idx in range(64):
            inputs.append(f"past_ssm_states.{idx}")
        for idx in range(64):
            inputs.append(f"past_conv_states.{idx}")
        return inputs

    def get_output_names(self):
        outputs = ['logits']
        for idx in range(64):
            outputs.append(f"present_ssm_states.{idx}")
        for idx in range(64):
            outputs.append(f"present_conv_states.{idx}")
        return outputs
    
    def save_tokenizer(self, tokenizer, out_dir):
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log.error(f'tokenizer loading failed with {e}')

    def convert_stateful_ov(self):
        language_model = self.get_model()
        
        language_model.config.torchscript = True

        # llm_input = torch.rand(( 1, 1, 2560), dtype=torch.float32)
        # logits, ssm_states, conv_states = language_model(inputs_embeds=llm_input, use_cache=True, return_dict=False)
        # # breakpoint()

        past_ssm_states = [
                torch.rand(1, 5120, 16, dtype=torch.float32)
                for _ in range(64)
            ]
        past_conv_states = [
                torch.rand(1, 5120, 4, dtype=torch.float32)
                for _ in range(64)
            ]
    
        ov_model = ov.convert_model(
            language_model,
            example_input={
                "input_ids": torch.tensor([[187]]),
                "cache_position": torch.tensor([0, 1, 2, 3]),
                "past_ssm_states": past_ssm_states,
                "past_conv_states": past_conv_states,
             },
        )

        # print('mamba_model inputs: ', ov_model.inputs)
        # print('mamba_model outputs: ', ov_model.outputs)

        for input, input_name in zip(ov_model.inputs, self.get_past_input_names()):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})
        
        shapes = {}
        for item in ov_model.inputs:
            if "past_ssm" in item.names.pop():
                  shapes[item] = item.partial_shape
                  shapes[item][1] = 5120
                  shapes[item][2] = 16
            if "past_conv" in item.names.pop():
                  shapes[item] = item.partial_shape
                  shapes[item][1] = 5120
                  shapes[item][2] = 4
        ov_model.reshape(shapes)
        ov_model.validate_nodes_and_infer_types()

        print('mamba_model inputs: ', ov_model.inputs)
        print('mamba_model outputs: ', ov_model.outputs)

        # breakpoint()
        patch_stateful_ssm(ov_model)

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_stateful.xml"))
        self.save_tokenizer(self.tokenizer, self.ov_model_path)
        self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1,
            }
            ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(ov_compressed_model, Path(f"{self.ov_model_path}/llm_stateful_int4.xml"))

class LlmEmbdModel():
    def __init__(
        self,
        model=None,
        ov_model_path=None,
        device='CPU',
        fp16=False,
    ):
        self.name = "Mamba LLM Embd Model"
        self.model = model
        self.device=device
        self.ov_model_path = ov_model_path
        self.fp16=fp16
        self.inputs_dict = {}

    def get_model(self):
        return self.model.backbone.embeddings

    def get_input_names(self):
        inputs = ['input_ids']
        return inputs

    def get_output_names(self):
        outputs = ['inputs_embeds']
        return outputs

    def get_dynamic_axes(self):
        pass

    def get_sample_input(self):
            pass

    def convert_sdpa_ov(self):
        embd_model = self.get_model()        

        input_ids = torch.randint(0, 32020, ( 1, 3408))

        ov_model = ov.convert_model(
            embd_model,
            example_input={
                "input":  input_ids,
             },
        )
        # breakpoint()
        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})
        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/llm_embd.xml"))

class Mamba_OV:
    def __init__(self, pretrained_model_path=None, model=None, tokenizer=None, ov_model_path='/tmp/Mamba_ov/', device='CPU', llm_int4_compress=False):

        if model is None and pretrained_model_path:        
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_path,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_path, 
                trust_remote_code=True
            )
        elif model and tokenizer and pretrained_model_path is None:
            self.model = model
            self.tokenizer = tokenizer

        self.int4_compress = llm_int4_compress

        self.llm_embed_model = LlmEmbdModel(model=self.model, ov_model_path=ov_model_path, device=device)
        self.llm_stateful_model = MambaModel(model=self.model, tokenizer= self.tokenizer, ov_model_path=ov_model_path, device=device, int4_compress=self.int4_compress)

    def export_vision_to_ov(self):
        self.llm_embed_model.convert_sdpa_ov()
        self.llm_stateful_model.convert_stateful_ov()

class OVMambaForCausalLM(GenerationMixin):
    def __init__(
        self,
        core=None,
        ov_model_path=None,
        device='CPU',
        int4_compress=False,
        llm_infer_list=[],
    ):
        self.ov_model_path = ov_model_path
        self.core = core
        self.ov_device = device
        self.int4_compress = int4_compress

        if int4_compress:
            self.llm_model = core.read_model(Path(f"{ov_model_path}/lm_stateful_int4.xml"))
            self.llm_compiled_model = core.compile_model(self.llm_model, device)
        else:
            self.llm_model = core.read_model(Path(f"{ov_model_path}/llm_stateful.xml"))
            self.llm_compiled_model = core.compile_model(self.llm_model, device)
            
        self.llm_request = self.llm_compiled_model.create_infer_request()
        # self.compiled_model = core.compile_model(self.model, device, config = {'INFERENCE_PRECISION_HINT': 'f32'})

        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.llm_model.inputs)}
        self.output_names = {idx: key for idx, key in enumerate(self.llm_model.outputs)}
        self.config = AutoConfig.from_pretrained(ov_model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.device = torch.device("cpu")
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.past_len = None
        self.next_beam_idx = None
        self.main_input_name = "input_ids"
        self._supports_cache_class = False

        self.llm_embd = core.read_model(Path(f"{ov_model_path}/llm_embd.xml"))
        self.llm_embd_compiled_model = core.compile_model(self.llm_embd, device)
        self.llm_embd_request = self.llm_embd_compiled_model.create_infer_request()
        
        self.tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True)

        self.llm_infer_list = llm_infer_list

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True
    
    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values
    
    def llm_embd_run(self, input_ids):
        llm_embd_inputs = {}
        llm_embd_inputs['input_ids'] = input_ids

        self.llm_embd_request.start_async(llm_embd_inputs, share_inputs=True)
        self.llm_embd_request.wait()

        return torch.from_numpy(self.llm_embd_request.get_tensor("inputs_embeds").data)

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids,
            inputs_embeds,
            attention_mask,
            past_key_values,
            position_ids,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """General inference method"""
        inputs_dict = {}
        if past_key_values is not None:
            inputs_embeds = self.llm_embd_run(input_ids)
            inputs_dict['inputs_embeds'] = inputs_embeds
        else:
            self.past_len = 0
            self.llm_request.reset_state()
            inputs_dict['inputs_embeds'] = inputs_embeds

        batch_size = inputs_embeds.shape[0]
        if "beam_idx" in self.input_names:
            inputs_dict["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

        # print("inputs_dict: ", inputs_dict)
        # print(self.input_names)
        start = time.perf_counter()
        self.llm_request.start_async(inputs_dict, share_inputs=True)
        self.llm_request.wait()
        end = time.perf_counter()

        generation_time = (end - start) * 1000
        self.llm_infer_list.append(generation_time)

        past_key_values = ((),)
        self.past_len += inputs_dict["inputs_embeds"].shape[1]

        print('logits: ', self.llm_request.get_tensor("logits").data)
        return CausalLMOutputWithPast(
            loss=None,
            logits=torch.from_numpy(self.llm_request.get_tensor("logits").data),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # only last token for inputs_ids if the state is passed along.
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
            }
        )
       
        return model_inputs

    def get_input_embeds(self, input_ids=None):
        input_embeds = self.llm_embd_run(input_ids)
        
        return input_embeds


