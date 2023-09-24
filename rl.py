import pfrl
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import sys
import pandas as pd
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
import wandb
from tqdm import tqdm

from typing import List
import torch
# from accelerate import Accelerator
from pydantic import BaseModel

class RewardInput(BaseModel):
    prompt: str
    responses: List[str]


class ResponseModel(BaseModel):
    completion: str
    is_success: bool


class Item(BaseModel):
    roles: List[str]
    messages: List[str]
    successful_completions: List[str]

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

checkpoint = "robertmyers/targon-7b"
revision = "v1.1.8"

# accelerator = Accelerator()
# device = accelerator.device

# dataset
df = pd.read_parquet("./data/aa.parquet")

observation_list = [{"input": prompt} for prompt in df["prompt"]]

print( df["prompt"])

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=revision, torch_dtype="auto")

model.to("cuda")

class MyRLEnv(TextRLEnv):
    def __init__(self, model, tokenizer, observation_input, max_length, compare_sample, **kwargs):
        super().__init__(model, tokenizer, observation_input, max_length, compare_sample, **kwargs)
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        self.dpo_weight: float = 0.3
        self.rlhf_weight: float = 0.4
        self.reciprocate_weight: float = 0.3

        self.model = model
        self.tokenizer = tokenizer

        self.reward_functions = [
            OpenAssistantRewardModel(device=self.device),
            ReciprocateRewardModel(device=self.device),
            DirectPreferenceRewardModel(device=self.device),
        ]

        self.reward_weights = [
            self.rlhf_weight,
            self.reciprocate_weight,
            self.dpo_weight,
        ]


        self.blacklist = (
            Blacklist()
        )

        self.task_validator = (
            TaskValidator()
        )

        self.relevance_model = (
            RelevanceRewardModel(device=self.device)
        )

        # self.diversity_model = (
        #     DiversityRewardModel(device=self.device)
        # )

        self.nsfw_model = (
            NSFWRewardModel(device=self.device)
        )

        self.masking_functions = [
            self.blacklist,
            self.task_validator,
            self.relevance_model,
            # self.diversity_model,
            self.nsfw_model
        ]
        
    def compute_rewards(self, prompt: str, responses: List[str]) -> torch.FloatTensor:
        name = "augment"
        
        # Compute the rewards for the responses given the prompt.
        rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(self.device)
        for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
            reward_i, reward_i_normalized = reward_fn_i.apply(prompt, responses, name)
            rewards += weight_i * reward_i_normalized.to(self.device)

        for masking_fn_i in self.masking_functions:
            mask_i, mask_i_normalized = masking_fn_i.apply(prompt, responses, name)
            rewards *= mask_i_normalized.to(self.device)  # includes diversity

        return rewards

    def format_response(self, responses: List[str]) -> List[ResponseModel]:
        data = [ResponseModel(completion=r, is_success=True) for r in responses]
        return data

    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        total_reward = []
        output = ""

        prompt = RewardInput(prompt=input_item['input'], responses=predicted_list)
        responses = self.format_response(predicted_list)
        # rewards = self.reward_model.get_rewards(input_item['input'], predicted_list, "test")
        rewards = self.compute_rewards(prompt, responses)
        
        if finish:
            reward = [1] * len(predicted_list)  # calculate reward score base on predicted_list
        return list(rewards.cpu())




# observaton_list = [{"input":"explain how attention work in seq2seq model"}]
# env = MyRLEnv(model, tokenizer, observation_input=observation_list, max_length=2048, compare_sample=2)
env = MyRLEnv(model, tokenizer, observation_list, 2048, 1, unfreeze_layer_from_past=4)
actor = TextRLActor(env, model, tokenizer,
                    act_deterministically=False,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,)
agent = actor.agent_ppo(update_interval=10, minibatch_size=3000, epochs=10)

# env, actor, agent, observation_list, model = accelerator.prepare(env, actor, agent, observation_list, model)
# print(observation_list[0]['input'])
# print(actor.predict(observation_list[0]))

n_episodes = 10000
max_episode_len = 200  # max sentence length


run = wandb.init(
    # Set the project where this run will be logged
    project="text-rl",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 3e-6,
        "epochs": 10,
        "n_episodes": n_episodes
    })

mean_reward = []

for i in tqdm(range(1, n_episodes + 1)):
    obs = env.reset()
    R = 0
    t = 0
    while True:
        action = agent.act(obs)
        obs, reward, done, pred = env.step(action)
        R += sum(reward)
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            mean_reward.append(R)
            mr = sum(mean_reward) / len(mean_reward)
            wandb.log({"reward": R, "mean_reward": mr})
            wandb.Table(columns=["response", "score", "step"], data=[[pred['predicted_str'][0], R, i]])
            break
    
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
        stats = agent.get_statistics()
        statistics_dict = dict(statistics)
        avg_value = statistics_dict['average_value']
        average_entropy = statistics_dict['average_entropy']
        average_value_loss = statistics_dict['average_value_loss']
        average_policy_loss = statistics_dict['average_policy_loss']
        n_updates = statistics_dict['n_updates']
        explained_variance = statistics_dict['explained_variance']
        wandb.log({
            "avg_value": avg_value,
            "average_entropy": average_entropy,
            "average_value_loss": average_value_loss,
            "average_policy_loss": average_policy_loss,
            "n_updates": n_updates,
            "explained_variance": explained_variance
        })
    
    if i % 250 == 0:
        outdir = "./output"
        suffix = ""
        dirname = os.path.join(outdir, "{}{}".format(t, suffix))
        agent.save(dirname)

print(actor.predict(observation_list[0]))