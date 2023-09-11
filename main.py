import hydra
import torch
from torch.optim import AdamW
from models.prompt_ast import PromptAST
from models.prompt_w2v import PromptW2V
from fluentspeech import FluentSpeech
from transformers import AutoFeatureExtractor
from torch.utils.data import DataLoader
import soundfile
import wandb
from tqdm import tqdm

# CREATES SPECTROGRAMS
def data_processing(data, processor, wav2vec=False):
    y = [] 
    x = []

    for i in range(len(data)):
        waveform, sample_rate = soundfile.read(data[i][0])
        if wav2vec:
            spec = processor(waveform, sampling_rate=sample_rate, padding="max_length", max_length=64000, truncation=True, return_tensors='pt')
        else:
            spec = processor(waveform, sampling_rate=sample_rate, return_tensors='pt')
        x.append(spec['input_values'])
        # intent
        intent = data[i][1]
        y.append(torch.tensor(intent))
    
    return torch.cat(x), torch.tensor(y)

# EPOCH TRAINING GRADIENT ACCUMULATOR
def train_one_epoch_acc(model, train_loader, loss_fn, epoch_index, optimizer, device, accum_iter=2):
    running_loss = 0.
    last_loss = 0.
    total = 0.
    accuracy = 0.
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Make predictions for this batch
        outputs = model(inputs)
        # Argmax on predictions
        _, predictions = torch.max(outputs, 1)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        # normalize loss to account for batch accumulation
        loss = loss / accum_iter 

        loss.backward()

        # Accuracy Computation
        total += labels.size(0)
        accuracy += (predictions == labels).sum().item()

        # weights update
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
            # print(f"backward {i}")
            optimizer.step()
            optimizer.zero_grad()


        # Gather data and report
        running_loss += loss.item()
        if i % 200 == 199:
            last_loss = running_loss / 200 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    
    intent_accuracy_train = (100 * accuracy / total)


    return last_loss, intent_accuracy_train

# EPOCH TRAINING
def train_one_epoch(model, train_loader, loss_fn, epoch_index, optimizer, device):
    running_loss = 0.
    last_loss = 0.
    total = 0.
    accuracy = 0.
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        # Argmax on predictions
        _, predictions = torch.max(outputs, 1)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Accuracy Computation
        total += labels.size(0)
        accuracy += (predictions == labels).sum().item()
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 200 == 199:
            last_loss = running_loss / 200 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    
    intent_accuracy_train = (100 * accuracy / total)


    return last_loss, intent_accuracy_train

def evaluate():
    pass

@hydra.main(version_base=None, config_path='config', config_name='args')
def main(args):

    # WANDB
    if args.USE_WANDB:
        
        wandb.init(project=args.PROJECT_NAME, name=args.EXP_NAME,entity="sciapponi", config = args)

    # VARIABLE DEFINITION
    data_path = args.DATA_PATH
    model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"

    if args.WAV2VEC:
        model_ckpt = "facebook/wav2vec2-base-960h"
    device = torch.device(args.DEVICE)
    torch.cuda.set_device(0)
    torch.set_num_threads(20)
    # AST Processor which computes spectrograms
    MAX_LENGTH= args.MAX_LENGTH
    max_len_audio = 64000
    processor = AutoFeatureExtractor.from_pretrained(model_ckpt, max_length=MAX_LENGTH)
    prompt_config = args.PROMPT
    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    NUM_WORKERS = args.NUM_WORKERS

    print("Loading Data")
    # DATASETS
    train_data = FluentSpeech(data_path,train="train", max_len_audio=64000)
    test_data = FluentSpeech(data_path,train="test", max_len_audio=64000)
    val_data = FluentSpeech(data_path,train="valid", max_len_audio=64000)

    # DATA LOADERS
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: data_processing(x, processor = processor, wav2vec=args.WAV2VEC), pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: data_processing(x, processor = processor, wav2vec=args.WAV2VEC), pin_memory=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: data_processing(x, processor = processor, wav2vec=args.WAV2VEC), pin_memory=True, num_workers=NUM_WORKERS)

    # MODEL DEFINITION
    if args.WAV2VEC:
        model = PromptW2V(prompt_config=prompt_config, max_length=MAX_LENGTH, model_ckpt=model_ckpt, num_classes=31).to(device)
        # REQUIRES_GRAD_ = FALSE
        model.encoder.requires_grad_(False)
        model.feature_extractor.requires_grad_(False)
        model.feature_projection.requires_grad_(False)
    else:
        model = PromptAST(prompt_config=prompt_config, max_length=MAX_LENGTH, model_ckpt=model_ckpt, num_classes=31).to(device)
        # REQUIRES_GRAD_ = FALSE
        model.encoder.requires_grad_(False)
        model.embeddings.requires_grad_(False)
    print(model)

    
    
    # PRINT MODEL PARAMETERS
    n_parameters = sum(p.numel() for p in model.parameters())
    print('Number of params of the model:', n_parameters)
    n_parameters = sum(p.numel() for p in model.encoder.parameters())
    print('Number of params of the encoder:', n_parameters)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params of the model:', n_parameters)

    # OPTIMIZER and LOSS DEFINITION
    optimizer = AdamW(model.parameters(),lr=args.LR,betas=(0.9,0.98),eps=1e-6,weight_decay=args.WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.LABEL_SMOOTHING)
    last_lr = args.LR
    # LR SCHEDULER
    if args.USE_SCHEDULER:
        T_0 = EPOCHS
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=0.0005)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*EPOCHS)

    # TRAINING LOOP
    epoch_number = 0
    
    best_vloss = 1_000_000.
    print("TRAINING STARTED")
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
    
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        if args.GRADIENT_ACC:
            avg_loss, intent_accuracy_train = train_one_epoch_acc(model=model, 
                                                            train_loader=train_loader,
                                                            loss_fn=loss_fn,
                                                            epoch_index=epoch,
                                                            optimizer=optimizer,
                                                            device=device,
                                                            accum_iter=16//BATCH_SIZE
                                                            )
        else:
            avg_loss, intent_accuracy_train = train_one_epoch(model=model, 
                                                            train_loader=train_loader,
                                                            loss_fn=loss_fn,
                                                            epoch_index=epoch,
                                                            optimizer=optimizer,
                                                            device=device
                                                            )
        
        # Learning rate scheduler step
        if args.USE_SCHEDULER:
            scheduler.step(epoch)
            last_lr = scheduler.get_last_lr()
            print(f"Learning rate at epoch {epoch}: {last_lr}")
    
        running_tloss = 0.0
        total = 0.
        taccuracy=0.
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        print("TEST:")
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, tdata in tqdm(enumerate(test_loader), total=len(test_loader)):
                tinputs, tlabels = tdata
                tinputs = tinputs.to(device)
                tlabels = tlabels.to(device)
                toutputs = model(tinputs)
                # Argmax on predictions
                _, tpredictions = torch.max(toutputs, 1)

                tloss = loss_fn(toutputs, tlabels)
                running_tloss += tloss
                # Accuracy Computation
                total += tlabels.size(0)
                taccuracy += (tpredictions == tlabels).sum().item()

        intent_accuracy_test = (100 * taccuracy / total)
        avg_tloss = running_tloss / (i + 1)

        running_vloss = 0.0
        total = 0.
        vaccuracy = 0.
        print("VALIDATION:")
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(val_loader), total=len(val_loader)):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                # Argmax on predictions
                _, vpredictions = torch.max(voutputs, 1)

                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                # Accuracy Computation
                total += vlabels.size(0)
                vaccuracy += (vpredictions == vlabels).sum().item()

        intent_accuracy_val = (100 * vaccuracy / total)
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} traing {} valid {}'.format(avg_loss, avg_tloss, avg_vloss))
    
        #Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'saved_models/{args.EXP_NAME}_model_{epoch_number}'
            torch.save(model.state_dict(), model_path)
    
        epoch_number += 1

        # WANDB LOGS
        if args.USE_WANDB:
            wandb.log({"train_loss": avg_loss, 
                       "test_loss": avg_tloss, 
                       "valid_loss": avg_vloss,
                       "intent_accuracy_train": intent_accuracy_train,
                       "intent_accuracy_test": intent_accuracy_test,
                       "intent_accuracy_val": intent_accuracy_val,
                       "lr": last_lr
                       }
                      )

    if args.USE_WANDB:
        wandb.finish()


if __name__=="__main__":
    main()