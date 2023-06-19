import hydra
import torch
from models.prompt_ast import PromptAST
from fluentspeech import FluentSpeech
from transformers import AutoFeatureExtractor
from torch.utils.data import DataLoader
import soundfile
import wandb

# CREATES SPECTROGRAMS
def data_processing(data, processor):
    y = [] 
    x = []

    for i in range(len(data)):
        waveform, sample_rate = soundfile.read(data[i][0])
        spec = processor(waveform, sampling_rate=sample_rate, return_tensors='pt')
        x.append(spec['input_values'])
        # intent
        intent = data[i][1]
        y.append(torch.tensor(intent))

    return torch.cat(x), torch.tensor(y)

def train_one_epoch(model, training_loader, epoch_index, optimizer, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def evaluate():
    pass

@hydra.main(version_base=None, config_path='config', config_name='args')
def main(args):

    # VARIABLE DEFINITIONS
    data_path ='/home/ste/Datasets/'
    model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"
    processor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    prompt_config = args.PROMPT

    # DATASETS
    train_data = FluentSpeech(data_path,train="train", max_len_audio=64000)
    test_data = FluentSpeech(data_path,train="test", max_len_audio=64000)
    val_data = FluentSpeech(data_path,train="valid", max_len_audio=64000)

    # DATA LOADERS
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=lambda x: data_processing(x, processor = processor))
    test_loader = DataLoader(test_data, batch_size=4, shuffle=True, collate_fn=lambda x: data_processing(x, processor = processor))
    val_loader = DataLoader(val_data, batch_size=4, shuffle=True, collate_fn=lambda x: data_processing(x, processor = processor))

    # model definition
    model = PromptAST(prompt_config=prompt_config, model_ckpt=model_ckpt)
    print(model)
    # for i in range(epochs):

    #     train_one_epoch()


if __name__=="__main__":
    main()