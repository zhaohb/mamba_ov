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
import numpy as np
from dataclasses import dataclass

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
        return [ 'input_ids',]
    
    def get_past_input_names(self):
        inputs = [ 'input_ids']
        for idx in range(64):
            inputs.append(f"present.{idx}.ssm_states_input")
        for idx in range(64):
            inputs.append(f"present.{idx}.conv_states_input")
        return inputs

    def get_output_names(self):
        outputs = ['lm_logits']
        for idx in range(64):
            outputs.append(f"present.{idx}.ssm_states_output")
        for idx in range(64):
            outputs.append(f"present.{idx}.conv_states_output")
        return outputs
    
    def save_tokenizer(self, tokenizer, out_dir):
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log.error(f'tokenizer loading failed with {e}')

    def convert_ov(self):
        language_model = self.get_model()
        
        language_model.config.torchscript = True
        ov_model = ov.convert_model(
            language_model,
            example_input={
                "input_ids": torch.tensor([[8262,  849,  403,  368, 2509,   32]]),
             },
        )

        print('mamba_model inputs: ', ov_model.inputs)
        print('mamba_model outputs: ', ov_model.outputs)

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/ov_mamba.xml"))
        self.save_tokenizer(self.tokenizer, self.ov_model_path)
        self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1,
            }
            ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(ov_compressed_model, Path(f"{self.ov_model_path}/ov_mamba_int4.xml"))

    def convert_past_ov(self):
        language_model = self.get_model()
        
        language_model.config.torchscript = True

        logits, ssm_states, conv_states = language_model(input_ids = torch.tensor([[8262,  849,  403,  368, 2509,   32]]), use_cache=True, return_dict=False)
        ov_model = ov.convert_model(
            language_model,
            example_input={
                "input_ids": torch.tensor([[187]]),
                "ssm_states": ssm_states,
                "conv_states": conv_states,
             },
        )

        print('mamba_model inputs: ', ov_model.inputs)
        print('mamba_model outputs: ', ov_model.outputs)

        for input, input_name in zip(ov_model.inputs, self.get_past_input_names()):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, self.get_output_names()):
            output.get_tensor().set_names({output_name})

        ov.save_model(ov_model, Path(f"{self.ov_model_path}/ov_past_mamba.xml"))
        self.save_tokenizer(self.tokenizer, self.ov_model_path)
        self.model.config.save_pretrained(self.ov_model_path)

        if self.int4_compress:
            compression_configuration = {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1,
            }
            ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
            ov.save_model(ov_compressed_model, Path(f"{self.ov_model_path}/ov_past_mamba_int4.xml"))

@dataclass
class OvMambaCausalLMOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    logits: Optional[torch.FloatTensor] = None
    ssm_states: Optional[List[torch.FloatTensor]] = None
    conv_states: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

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
            self.llm_past_model = core.read_model(Path(f"{ov_model_path}/ov_past_mamba_int4.xml"))
            self.llm_past_compiled_model = core.compile_model(self.llm_past_model, device)
            self.llm_model = core.read_model(Path(f"{ov_model_path}/ov_mamba_int4.xml"))
            self.llm_compiled_model = core.compile_model(self.llm_model, device)
        else:
            self.llm_past_model = core.read_model(Path(f"{ov_model_path}/ov_past_mamba.xml"))
            self.llm_past_compiled_model = core.compile_model(self.llm_past_model, device)
            self.llm_model = core.read_model(Path(f"{ov_model_path}/ov_mamba.xml"))
            self.llm_compiled_model = core.compile_model(self.llm_model, device)
            
        self.llm_request = self.llm_compiled_model.create_infer_request()
        self.llm_past_request = self.llm_past_compiled_model.create_infer_request()
        # self.compiled_model = core.compile_model(self.model, device, config = {'INFERENCE_PRECISION_HINT': 'f32'})

        self.config = AutoConfig.from_pretrained(ov_model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.device = torch.device("cpu")
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.past_len = None
        self.main_input_name = "input_ids"
        self._supports_cache_class = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True)

        self.llm_infer_list = llm_infer_list

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True
    
    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        ssm_states: Optional[List[torch.FloatTensor]] = None,
        conv_states: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> OvMambaCausalLMOutput:
        return self.forward(
            input_ids,
            inputs_embeds,
            attention_mask,
            ssm_states,
            conv_states,
            position_ids,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        ssm_states: Optional[List[torch.FloatTensor]] = None,
        conv_states: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> OvMambaCausalLMOutput:
        """General inference method"""
        inputs_dict = {}
        inputs_dict['input_ids'] = input_ids

        if ssm_states is not None:
            for idx in range(64):
                inputs_dict[f"present.{idx}.conv_states_input"] = conv_states[idx]
                inputs_dict[f"present.{idx}.ssm_states_input"] = ssm_states[idx]

        # print('input_ids: ', inputs_dict['input_ids'])
        start = time.perf_counter()
        if ssm_states is not None:
            self.llm_past_request.start_async(inputs_dict, share_inputs=True)
            self.llm_past_request.wait()
        else:
            self.llm_request.start_async(inputs_dict, share_inputs=True)
            self.llm_request.wait()
        end = time.perf_counter()

        generation_time = (end - start) * 1000
        self.llm_infer_list.append(generation_time)

        conv_states_list = []
        ssm_states_list = []
        for idx in range(64):
            if ssm_states is not None:
                conv_states_list.append(torch.from_numpy(self.llm_past_request.get_tensor(f"present.{idx}.conv_states_output").data))
                ssm_states_list.append(torch.from_numpy(self.llm_past_request.get_tensor(f"present.{idx}.ssm_states_output").data))
            else:
                conv_states_list.append(torch.from_numpy(self.llm_request.get_tensor(f"present.{idx}.conv_states_output").data))
                ssm_states_list.append(torch.from_numpy(self.llm_request.get_tensor(f"present.{idx}.ssm_states_output").data))

        logits = None
        if ssm_states is not None:
            logits = torch.from_numpy(self.llm_past_request.get_tensor("lm_logits").data)
        else:
            logits = torch.from_numpy(self.llm_request.get_tensor("lm_logits").data)
        
        return OvMambaCausalLMOutput(
            logits=logits,
            ssm_states=ssm_states_list,
            conv_states=conv_states_list,
            hidden_states=None,
        )   

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        model_kwargs['ssm_states'] = outputs['ssm_states']
        model_kwargs['conv_states'] = outputs['conv_states']

        return model_kwargs
    
    def prepare_inputs_for_generation(
        self, input_ids, 
        # cache_params: Optional[MambaCache] = None, 
        ssm_states: Optional[List[torch.FloatTensor]] = None,
        conv_states: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds=None, attention_mask=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if ssm_states is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and ssm_states is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
            
        if ssm_states is not None:
            model_inputs["ssm_states"] = ssm_states
            model_inputs["conv_states"] = conv_states
        return model_inputs


