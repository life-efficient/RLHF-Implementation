from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer
import torch
import pandas as pd


def loss_function(preferred_response_reward, alternate_response_reward):
    return -torch.mean(torch.log(torch.sigmoid(preferred_response_reward - alternate_response_reward)))


def create_response_pairs():

    data = pd.read_csv('reward_dataset.csv', sep="|")

    data = data.to_dict(orient="records")
    response_pairs = []

    for row in data:
        prompt = row["Prompt"]
        response_pairs.append(
            (prompt, row["Most preferable response"], row["Somewhat preferable response"]))
        response_pairs.append(
            (prompt, row["Most preferable response"], row["Least preferable response"]))
        response_pairs.append(
            (prompt, row["Somewhat preferable response"], row["Least preferable response"]))

    return response_pairs


class RewardDataset(torch.utils.data.Dataset):
    def __init__(self):
        """Initializes the dataset."""
        self.response_pairs = create_response_pairs()
        print("Number of response pairs:", len(self.response_pairs))

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.response_pairs)

    def __getitem__(self, idx):
        """Returns the example in the dataset at the given index."""

        # Get the response pair at the given index
        response_pair = self.response_pairs[idx]
        prompt, preferred_response, alternate_response = response_pair

        # Return the preferred response, alternate response
        return prompt, preferred_response, alternate_response


class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.regression_head = torch.nn.Linear(768, 1)
        self.to(self.device)

    def forward(self, context, response):
        """
        Returns a scalar value representing the reward for this response, given the context.
        Args:
            context (str): The context. aka. the prompt.
            response (str): The response. aka. the response to the prompt.
        Returns:
            float: The reward for generating this response given the context.    
        """

        entire_text = context + response
        context_dict = self.tokenizer(
            '<|startoftext|>' + entire_text + '<|endoftext|>',
            #    truncation=True,
            #    max_length=max_length,
            #    padding="max_length"
        )

        input_ids = torch.tensor(context_dict.input_ids)
        attention_mask = torch.tensor(context_dict.attention_mask)

        # Move to GPU
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Forward pass
        gpt2_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        all_output_vectors = gpt2_outputs.last_hidden_state
        last_output_vector = all_output_vectors[-1]

        # add batch_size dimension
        last_output_vector = last_output_vector.unsqueeze(0)
        reward = self.regression_head(last_output_vector)

        return reward


def train_and_save_reward_model(epochs=10):

    model = RewardModel()

    # Create the dataset and dataloader
    dataset = RewardDataset()

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, betas=(0.9, 0.95))  # as used in the InstructGPT paper

    # Set up logging
    writer = SummaryWriter()  # for logging our loss to TensorBoard
    # for setting the x-axis of our TensorBoard plots (loss vs. batch index)
    batch_idx = 0
    # Train the model
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for batch in tqdm(dataset):

            prompt, preferred_response, alternate_response = batch

            preferred_response_reward = model(prompt, preferred_response)
            alternate_response_reward = model(prompt, alternate_response)

            loss = loss_function(preferred_response_reward,
                                 alternate_response_reward)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            writer.add_scalar("Reward Model Loss/Train",
                              loss.item(), batch_idx)
            batch_idx += 1
            torch.save(model.state_dict(),
                       f"epoch-{epoch}-reward_model_params.pt")
    torch.save(model.state_dict(), "reward_model_params.pt")
