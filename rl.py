import pfrl
from textrl import TextRLEnv, TextRLActor, train_agent_with_evaluation
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import sys
import pandas as pd
from reward.open_assistant import OpenAssistantRewardModel

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

checkpoint = "robertmyers/targon-7b"
revision = "v1.1.8"



# dataset
df = pd.read_parquet("./data/aa.parquet")

observation_list = [{"input": prompt} for prompt in df["prompt"]][:100]

print( df["prompt"])

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=revision, torch_dtype="auto")

model.to("cuda")

class MyRLEnv(TextRLEnv):
    def __init__(self):
        self.reward_model = OpenAssistantRewardModel("cuda")

    def get_reward(self, input_item, predicted_list, finish):  # predicted will be the list of predicted token
        self.reward_model.get_rewards(input_item, predicted_list, "test")
        if finish:
            reward = [1]  # calculate reward score base on predicted_list
        return reward


# observaton_list = [{"input":"explain how attention work in seq2seq model"}]
env = TextRLEnv(model, tokenizer, observation_input=observation_list, max_length=256, compare_sample=2)
actor = TextRLActor(env, model, tokenizer,
                    act_deterministically=False,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,)
agent = actor.agent_ppo(update_interval=2, minibatch_size=2, epochs=10)
print(actor.predict(observation_list[0]))

train_agent_with_evaluation(
    agent,
    env,
    steps=100,
    eval_n_steps=None,
    eval_n_episodes=1,
    eval_interval=2,
    outdir='bloom—test',
)



print(actor.predict(observation_list[0]))