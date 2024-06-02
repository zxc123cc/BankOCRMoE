# transformers==4.30.2
import os
import sys
base_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(base_dir, '../'))
import torch.nn.functional as F
from transformers.generation.utils import *
from transformers.modeling_utils import PreTrainedModel

class EnsembleGenerateModel(PreTrainedModel):
    def __init__(self,config = None,model_list = None,weight_list = None):
        # super(EnsembleGenerateModel, self).__init__()
        super().__init__(config)
        self.model_list = model_list
        if weight_list is None:
            self.weight_list = [1.0/len(model_list)] * len(model_list)
        else:
            self.weight_list = weight_list

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        for i in range(len(self.model_list)):
            self.model_list[i]._validate_model_class()

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation -- update the generation config
            # model attribute accordingly, if it was created from the model config
            if self.model_list[0].generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.model_list[0].config)
                if new_generation_config != self.model_list[0].generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use a generation configuration file (see"
                        " https://huggingface.co/docs/transformers/main_classes/text_generation)"
                    )
                    self.model_list[0].generation_config = new_generation_config
            generation_config = self.model_list[0].generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        for i in range(len(self.model_list)):
            self.model_list[i]._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        model_kwargs_list = []
        for i in range(len(self.model_list)):
            inputs_tensor, model_input_name, tmp_model_kwargs = self.model_list[i]._prepare_model_inputs(
                inputs, generation_config.bos_token_id, model_kwargs
            )
            model_kwargs_list.append(tmp_model_kwargs)

        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        for i in range(len(self.model_list)):
            model_kwargs_list[i]["output_attentions"] = generation_config.output_attentions
            model_kwargs_list[i]["output_hidden_states"] = generation_config.output_hidden_states
            model_kwargs_list[i]["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.model_list[0].forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs_list[0]

        if model_kwargs_list[0].get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            for i in range(len(self.model_list)):
                model_kwargs_list[i]["attention_mask"] = self.model_list[i]._prepare_attention_mask_for_generation(
                    inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
                )

        # decoder-only models should use left-padding for generation
        if not self.model_list[0].config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                    generation_config.pad_token_id is not None
                    and len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.model_list[0].config.is_encoder_decoder and "encoder_outputs" not in model_kwargs_list[0]:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            for i in range(len(self.model_list)):
                model_kwargs_list[i] = self.model_list[i]._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, model_kwargs_list[i], model_input_name
                )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.model_list[0].config.is_encoder_decoder:
            for i in range(len(self.model_list)):
                input_ids, model_kwargs_list[i] = self.model_list[i]._prepare_decoder_input_ids_for_generation(
                    batch_size=batch_size,
                    model_input_name=model_input_name,
                    model_kwargs=model_kwargs_list[i],
                    decoder_start_token_id=generation_config.decoder_start_token_id,
                    bos_token_id=generation_config.bos_token_id,
                    device=inputs_tensor.device,
                )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs_list[0].pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.model_list[0].config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 7. determine generation mode
        is_constraint_gen_mode = (
                generation_config.constraints is not None or generation_config.force_words_ids is not None
        )

        is_contrastive_search_gen_mode = (
                (generation_config.num_beams == 1)
                and generation_config.top_k is not None
                and generation_config.top_k > 1
                and generation_config.do_sample is False
                and generation_config.penalty_alpha is not None
                and generation_config.penalty_alpha > 0
        )

        is_greedy_gen_mode = (
                (generation_config.num_beams == 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is False
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_sample_gen_mode = (
                (generation_config.num_beams == 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is True
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_beam_gen_mode = (
                (generation_config.num_beams > 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is False
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_beam_sample_gen_mode = (
                (generation_config.num_beams > 1)
                and (generation_config.num_beam_groups == 1)
                and generation_config.do_sample is True
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_group_beam_gen_mode = (
                (generation_config.num_beams > 1)
                and (generation_config.num_beam_groups > 1)
                and not is_constraint_gen_mode
                and not is_contrastive_search_gen_mode
        )
        is_assisted_gen_mode = False
        if assistant_model is not None:
            if not (is_greedy_gen_mode or is_sample_gen_mode):
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
            is_assisted_gen_mode = True

        if generation_config.num_beam_groups > generation_config.num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and generation_config.do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.model_list[0].device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.model_list[0].device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.model_list[0].device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        logits_processor = self.model_list[0]._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        # 9. prepare stopping criteria
        stopping_criteria = self.model_list[0]._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if is_assisted_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. If the assistant model is an encoder-decoder, prepare its encoder outputs
            if assistant_model.config.is_encoder_decoder:
                assistant_model_kwargs = copy.deepcopy(model_kwargs)
                inputs_tensor, model_input_name, assistant_model_kwargs = assistant_model._prepare_model_inputs(
                    inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_model_kwargs
                )
                assistant_model_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, assistant_model_kwargs, model_input_name
                )
                model_kwargs["assistant_encoder_outputs"] = assistant_model_kwargs["encoder_outputs"]

            # 12. run assisted generate
            return self.model_list[0].assisted_decoding(
                input_ids,
                assistant_model=assistant_model,
                do_sample=generation_config.do_sample,
                logits_processor=logits_processor,
                logits_warper=self.model_list[0]._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs_list[0],
            )
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )

            # 11. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            for i in range(len(self.model_list)):
                input_ids_tmp, model_kwargs_list[i] = self.model_list[i]._expand_inputs_for_generation(
                    input_ids=input_ids,
                    expand_size=generation_config.num_beams,
                    is_encoder_decoder=self.model_list[i].config.is_encoder_decoder,
                    **model_kwargs_list[i],
                )
            input_ids = input_ids_tmp
            # 13. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                model_kwargs_list=model_kwargs_list,
            )


    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            model_kwargs_list=None,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.model_list[0].generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.model_list[0].generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.model_list[0].generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.model_list[0].generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model_list[0].generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.model_list[0].generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.model_list[0].config.is_encoder_decoder:
            encoder_attentions = model_kwargs_list[0]["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs_list[0]["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            outputs_list = []
            for i in range(len(self.model_list)):
                model_inputs = self.model_list[i].prepare_inputs_for_generation(input_ids, **model_kwargs_list[i])

                outputs = self.model_list[i](
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                outputs_list.append(outputs)

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            for i in range(len(self.model_list)):
                # next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = outputs_list[i].logits[:, -1, :]
                # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
                # cannot be generated both before and after the `nn.functional.log_softmax` operation.
                # next_token_logits = self.model_list[i].adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
                if i == 0:
                    next_token_scores = nn.functional.log_softmax(
                        next_token_logits, dim=-1
                    ) * self.weight_list[i] # (batch_size * num_beams, vocab_size)
                else:
                    next_token_scores += nn.functional.log_softmax(
                        next_token_logits, dim=-1
                    ) * self.weight_list[i]

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.model_list[0].config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.model_list[0].config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model_list[0].config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            for i in range(len(self.model_list)):
                model_kwargs_list[i] = self.model_list[i]._update_model_kwargs_for_generation(
                    outputs, model_kwargs_list[i], is_encoder_decoder=self.model_list[i].config.is_encoder_decoder
                )
                if model_kwargs_list[i]["past_key_values"] is not None:
                    model_kwargs_list[i]["past_key_values"] = self.model_list[i]._reorder_cache(model_kwargs_list[i]["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.model_list[0].config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]