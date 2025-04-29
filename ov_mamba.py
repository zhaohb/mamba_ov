# from transformers import MambaConfig, OvMambaForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
import openvino as ov
from pathlib import Path
import nncf
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.utils import ModelOutput
from transformers.cache_utils import MambaCache
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

    # def get_input_names(self):
    #     return [ 'inputs_embeds',]
    
    def get_input_names(self):
        inputs = [ 'input_ids', 'cache_position']
        for idx in range(self.model.config.num_hidden_layers):
            inputs.append(f"past_ssm_states.{idx}")
        for idx in range(self.model.config.num_hidden_layers):
            inputs.append(f"past_conv_states.{idx}")
        return inputs

    def get_output_names(self):
        outputs = ['logits']
        for idx in range(self.model.config.num_hidden_layers):
            outputs.append(f"present_ssm_states.{idx}")
        for idx in range(self.model.config.num_hidden_layers):
            outputs.append(f"present_conv_states.{idx}")
        return outputs
    
    def save_tokenizer(self, tokenizer, out_dir):
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as e:
            log.error(f'tokenizer loading failed with {e}')

    def convert_stateful_ov(self):
        language_model = self.get_model()
        
        # language_model.config.torchscript = True

        past_ssm_states = [
                torch.rand(2, 5120, 16, dtype=torch.float32)
                for _ in range(self.model.config.num_hidden_layers)
            ]
        past_conv_states = [
                torch.rand(2, 5120, 4, dtype=torch.float32)
                for _ in range(self.model.config.num_hidden_layers)
            ]
    
        input_ids = torch.tensor([[41328, 27042,  1230,  4217, 11686, 24707, 27775, 34687, 39179, 42198,
                                    40809,  4124, 36307, 28789, 21264,  6422],
                                  [26637,  5484, 29898,  1557, 31845,  1967,  2883, 49649, 17155,  1341,
                                    43345,  1232, 32996, 48297,   905, 25175]])
        cache_position = torch.tensor([ 6, 15, 14,  1])
        # breakpoint()
        ov_model = ov.convert_model(
            language_model,
            example_input={
                "input_ids": input_ids,
                "cache_position": cache_position,
                "past_ssm_states": past_ssm_states,
                "past_conv_states": past_conv_states,
             },
        )

        # print('mamba_model inputs: ', ov_model.inputs)
        # print('mamba_model outputs: ', ov_model.outputs)

        for input, input_name in zip(ov_model.inputs, self.get_input_names()):
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

class OVMambaCache(MambaCache):
    def __init__(
        self,
        config: "PretrainedConfig",
        batch_size: int = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[torch.device, str]] = None,
        max_batch_size: Optional[int] = None,
        conv_states: Optional[List[torch.Tensor]] = None,
        ssm_states: Optional[List[torch.Tensor]] = None,
    ):
        self.dtype = dtype
        self.max_batch_size = batch_size or max_batch_size
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        if conv_states is not None:
            self.conv_states = conv_states
        else:
            self.conv_states = []
            for _ in range(config.num_hidden_layers):
                conv_state: torch.Tensor = torch.zeros(
                    self.max_batch_size, self.intermediate_size, self.conv_kernel_size, device=self.device, dtype=dtype
                )
                self.conv_states.append(conv_state)

        if ssm_states is not None:
            self.ssm_states = ssm_states
        else:
            self.ssm_states: List[torch.Tensor] = []
            for _ in range(config.num_hidden_layers):
                ssm_state: torch.Tensor = torch.zeros(
                    self.max_batch_size,
                    self.intermediate_size,
                    self.ssm_state_size,
                    device=self.device,
                    dtype=dtype,
                )

                self.ssm_states.append(ssm_state)


@dataclass
class MambaOutput(ModelOutput):
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
    cache_params: Optional[OVMambaCache] = None
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
            self.llm_model = core.read_model(Path(f"{ov_model_path}/llm_stateful_int4.xml"))
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(ov_model_path, trust_remote_code=True)

        self.llm_infer_list = llm_infer_list

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        cache_params=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MambaOutput:
        return self.forward(
            input_ids,
            attention_mask,
            cache_params,
            use_cache,
            cache_position,
            **kwargs,
        )
    
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            cache_params=None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            inputs = {"input_ids": input_ids.cpu().numpy()}
            if "cache_position" in self.input_names:
                inputs["cache_position"] = cache_position.cpu().numpy()
            if "attention_mask" in self.input_names:
                inputs["attention_mask"] = cache_position.cpu().numpy()

            if cache_params is None:
                # This is the first iteration in a sequence, reset all states
                if self.llm_request is not None:
                    self.llm_request.reset_state()
                self._past_length = 0

            ssm_states, conv_states = [], []
            self.llm_request.start_async(inputs, share_inputs=True)
            self.llm_request.wait()
            logits = torch.from_numpy(self.llm_request.get_tensor("logits").data)
            print("logits: ", logits)


            self._past_length += input_ids.shape[1]
            cache_params = OVMambaCache(self.config, input_ids.shape[0], conv_states=conv_states, ssm_states=ssm_states)

            return MambaOutput(logits=logits, cache_params=cache_params)

    def _update_model_kwargs_for_generation(
        self, 
        outputs: ModelOutput, 
        model_kwargs: Dict[str, Any], 
        num_new_tokens: int = 1, 
        **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Overwitten -- uses `cache_params` as opposed to `past_key_values`

        if use_cache:
            # `cache_position` should have been initialized in `generate`
            if cache_position is None:
                raise ValueError(
                    "`cache_position` should not be None as it should have been initialized in "
                    "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                    "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
                )
            if cache_position[0] > 0:
                input_ids = input_ids[:, -1].unsqueeze(-1)

                if attention_mask is not None:
                    attention_mask = None

            else:
                # we initialize the `cache_position` to full size of `conv_states` at prefill stage
                # considering padding will be applied when input length is shorter, and truncation
                # will be applied when it is longer, so it will be equivalent to always have it match
                # the length of `cache_params.conv_states`, which is `config.conv_kernel`
                cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "cache_params": cache_params,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "attention_mask": attention_mask,
            }
        )
        print("model_inputs: ", model_inputs)
        return model_inputs



