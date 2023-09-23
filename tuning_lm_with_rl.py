import os

import torch
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline
)

from typing import List
import time
import threading
import requests

from bittensor import logging as logger

from reward import (
    Blacklist,
    NSFWRewardModel,
    OpenAssistantRewardModel,
    ReciprocateRewardModel,
    RelevanceRewardModel,
    MockRewardModel,
    DahoasRewardModel,
    DirectPreferenceRewardModel,
    TaskValidator,
    DiversityRewardModel,
    PromptRewardModel,
    RewardModelType,
)

from pydantic import BaseModel

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()

class RewardInput(BaseModel):
    prompt: str
    responses: List[str]


class ResponseModel(BaseModel):
    completion: str
    is_success: bool



@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="robertmyers/gigatargon", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=1024, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.02, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="./checkpoints/tuning_llama_rl/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    revision: Optional[str] = field(default="main", metadata={"help": "the git revision"})
    dataset_revision: Optional[str] = field(default="main", metadata={"help": "the dataset revision"})


parser = HfArgumentParser(ScriptArguments)  
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

set_seed(script_args.seed)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
        tokenizer, dataset_name, input_min_text_length=2, input_max_text_length=8
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """

    train_dataset = load_dataset(dataset_name, split="train[:5%]", revision=script_args.dataset_revision)
    original_columns = train_dataset.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            # "query": [],
            "input_ids": [],
        }
        for prompt in examples["prompt"]:
            tokenized_sample = tokenizer(prompt, truncation=True)
            # new_examples["query"].append(chosen)
            new_examples["input_ids"].append(tokenized_sample["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.max_length, batched=False)

    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


reward_model_name = script_args.reward_model_name
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
rw_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True
}

if "decapoda" in script_args.model_name.lower():
    tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name, revision=script_args.revision)
    # required for llama
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, revision=script_args.revision)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(tokenizer, script_args.dataset_name)

# Now let's build the model, the reference model, and the tokenizer.
current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
    revision=script_args.revision,
)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug


dpo_weight: float = 1.0
rlhf_weight: float = 0.4
reciprocate_weight: float = 0.3
dahoas_weight: float = 0
prompt_based_weight: float = 0
name = "augment"

reward_weights = [
            rlhf_weight,
            # reciprocate_weight,
            # dpo_weight,
        ]

reward_functions = [
    OpenAssistantRewardModel(device=device),
    # ReciprocateRewardModel(device=device),
    # DirectPreferenceRewardModel(device=device),
]

# blacklist = (
#     Blacklist()
# )

# relevance_model = (
#     RelevanceRewardModel(device=device)
# )

# diversity_model = (
#     DiversityRewardModel(device=device)
# )

# nsfw_model = (
#     NSFWRewardModel(device=device)
# )

# task_validator = (
#     TaskValidator()
# )

# masking_functions = [
#     blacklist,
#     task_validator,
#     relevance_model,
#     diversity_model,
#     nsfw_model
# ]

def compute_rewards(prompt: str, responses: List[str]) -> torch.FloatTensor:
    name = "augment"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Compute the rewards for the responses given the prompt.
    rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(device)
    for weight_i, reward_fn_i in zip(reward_weights, reward_functions):
        reward_i, reward_i_normalized = reward_fn_i.apply(prompt, responses, name)
        rewards += weight_i * reward_i_normalized.to(device)
        logger.trace(str(reward_fn_i.name), reward_i_normalized.tolist())

    # for masking_fn_i in masking_functions:
    #     mask_i, mask_i_normalized = masking_fn_i.apply(prompt, responses, name)
    #     rewards *= mask_i_normalized.to(device)  # includes diversity
    #     logger.trace(str(masking_fn_i.name), mask_i_normalized.tolist())

    
    logger.info(f"Final reward: {rewards.tolist()}")  # Log final reward
    torch.cuda.empty_cache()
    return rewards

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "temperature": 0.7,
    "top_k": 0.0,
    "top_p": 0.99,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch['query'] = tokenizer.batch_decode(question_tensors, skip_special_tokens=True)
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # batch["response"] = ["testing" for _ in range(len(batch["query"]))]

    # logger.info('batch["query"]', batch["query"])
    # logger.info('batch["response"]', batch["response"])

    rewards_pre = []
    for prompt, response in zip(batch["query"], batch["response"]):
        if response != "":
            prompt = RewardInput(prompt=prompt, responses=[response])
            responses = [ResponseModel(completion=response, is_success=True)]


            reward_outputs = compute_rewards(prompt.prompt, responses)
            rewards_pre.append(reward_outputs[0])
        else:
            rewards_pre.append(torch.tensor(-1.0))
    rewards = [torch.tensor(output - script_args.reward_baseline) for output in rewards_pre]

    logger.info("reward", rewards_pre[0])
    logger.info('batch["response"]', batch["response"][0])

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
    torch.cuda.empty_cache()