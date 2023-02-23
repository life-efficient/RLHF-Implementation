from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


class SFTModel(GPT2LMHeadModel):
    def __init__(self):
        super().__init__()
        configuration = GPT2Config.from_pretrained(
            'gpt2', output_hidden_states=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            "gpt2", config=configuration)  # Load the tokenizer
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move the model to the GPU

    def forward(self, prompt, response):
        # Encode the data
        entire_text = prompt + response
        context_dict = self.tokenizer(
            '<|startoftext|>' + entire_text + '<|endoftext|>',
            #    truncation=True,
            #    max_length=max_length,
            #    padding="max_length"
        )

        input_ids = torch.tensor(context_dict.input_ids)
        labels = torch.tensor(context_dict.input_ids)
        attention_mask = torch.tensor(context_dict.attention_mask)

        # Move to GPU
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Run the model
        outputs = self(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        return outputs


class SFTDataset(torch.utils.data.Dataset):
    """Supervised Fine-Tuning Dataset

    Returns:
        prompt: str
        response: str
    """

    def __init__(self):
        with open("sft_dataset.json") as f:
            self.data = json.load(f)

    def __len__(self):
        """Defines the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Defines how to get a sample from the dataset by indexing it.

        Returns:
            prompt: str
            response: str
        """
        return self.data[idx]["prompt"], self.data[idx]["response"]


def train_and_save_SFT_model(epochs=10):

    # Create the model
    model = GPT2LMHeadModel.from_pretrained("gpt2")  # Load the model

    # Create the dataset and dataloader
    dataset = SFTDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Create the optimizer
    # as used in the InstructGPT paper
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, betas=(0.9, 0.95))

    # Set up logging
    writer = SummaryWriter()  # for logging our loss to TensorBoard
    # for setting the x-axis of our TensorBoard plots (loss vs. batch index)
    batch_idx = 0

    # Train the model
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        for batch in tqdm(dataloader):
            # Get the data
            prompt, response = batch
            prompt = prompt[0]
            response = response[0]

            # Forward pass
            outputs = model(prompt, response)

            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Zero the gradients
            optimizer.zero_grad()

            # Log the loss
            # print(f"Loss: {loss.item()}", batch_idx)
            writer.add_scalar("SFT Model Loss/train", loss.item(), batch_idx)
            batch_idx += 1
    torch.save(model.state_dict(), "sft_model_params.pt")
