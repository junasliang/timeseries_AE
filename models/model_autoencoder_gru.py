"""
Author: Junas
Model: Autoencoder with GRU cell.
Reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

# model.py
#import torch module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#import model summary modules
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

#import sklearn mse, mae
from sklearn.metrics import mean_squared_error, mean_absolute_error

#import helper function modules
from tqdm import tqdm
from random import random
import numpy as np
import sys

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p=0.1):
        super(Encoder,self).__init__()
        self.encoder_embed = nn.Linear(input_size,hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, encoder_inputs, input_lengths):
        encoder_inputs = self.encoder_embed(encoder_inputs)
        packed_inputs = pack_padded_sequence(encoder_inputs, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden
        # outputs shape: (batch_size, sequence length, hidden_size)
        # hidden shape:  (num_layers, batch_size, hidden_size)

# Decoder architecture without attention layer.
# Unfinished part, need extra modification.
class VanillaDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(VanillaDecoder, self).__init__()

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        MAX_LENGTH = encoder_outputs.size(1)

        decoder_input = torch.zeros((batch_size, 1, hidden_size), device=encoder_outputs.device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                decoder_input = decoder_output[:, :, :hidden_size].detach()  # detach from history and match input size

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden


# Attention layer.
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: (batch_size, num_layers, hidden_size)
        # keys: (batch_size, seq_len, hidden_size)
        query = query[:,-1,:].unsqueeze(1) #last layer of hidden
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # (batch_size, seq_len, 1)
        
        #check follow
        scores = scores.squeeze(2).unsqueeze(1)  # (batch_size, 1, seq_len)
        weights = F.softmax(scores, dim=-1)  # (batch_size, 1, seq_len)
        context = torch.bmm(weights,keys)  # (batch_size, 1, hidden_size)
        
        return context, weights


# Decoder with attention layer included
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.output_size = output_size
        self.decoder_embed = nn.Linear(output_size,hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size,num_layers, batch_first=True) #GRU input size is feature+hidden_size
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    # to do: try out different strategies of "initializing decoder input"
    # 1.last encoder output ===> shape mismatch during recursively training gru
    # 2.last encoder input + linear layer reshape
    # 3.Embedding layer to reshape encoder input first

    def forward(self, encoder_inputs, encoder_outputs, encoder_hidden, target_tensor=None, teacher_forcing_prob=0):
        # set initial decoder input as last step of encoder input (batch_size, 1, hidden_size)
        decoder_input = self.decoder_embed(encoder_inputs[:,-1,:].unsqueeze(1))
        batch_size = encoder_outputs.size(0)
        MAX_LENGTH = encoder_outputs.size(1)
        decoder_hidden = encoder_hidden

        decoder_outputs = []
        attentions = []

        #fix the len of for loop
        for i in range(MAX_LENGTH): # MAXLENGTH global or local is better?
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)
            
            tf = random() < teacher_forcing_prob
            if target_tensor is not None and  tf:
                # Teacher forcing: Feed the target as the next input
                decoder_input = self.decoder_embed(target_tensor[:,i,:].unsqueeze(1))  # Teacher forcing
                # print(f"decoder_output:{decoder_input.shape}")
            else:
                # Without teacher forcing: use its own predictions as the next input
                decoder_input = decoder_output.detach()  # detach from history as input. shape: (batch_size, 1, hidden_size)
                #print(f"notf:{decoder_input.shape}")

        decoder_outputs = torch.cat(decoder_outputs, dim=1) #(batch_size, seq_len, hidden)
        attentions = torch.cat(attentions, dim=1) #(batch_size, seq_len, seq_len)

        decoder_outputs = self.out(decoder_outputs)

        return decoder_outputs, decoder_hidden, attentions

    # Might need check output shape
    def forward_step(self, input, hidden, encoder_outputs):
        # input: (batch_size, 1, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size) the last hidden state of a cell
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        
        #query = hidden[-1].unsqueeze(1)  # hidden state of last layer (batch_size, 1, hidden_size)
        query = hidden.permute(1, 0, 2) #==> torch sample, shape: (batch_size, num_layers, hidden_size)
        context, attn_weights = self.attention(query, encoder_outputs)  # (batch_size, num_layers, hidden_size), (batch_size, 1, seq_len)

        input_gru = torch.cat((input, context), dim=2)  # (batch_size, 1, hidden_size * 2)
        output, hidden = self.gru(input_gru, hidden)  # (batch_size, 1, hidden_size)
        #output = self.out(output)  # (batch_size, 1, output_size)
        
        return output, hidden, attn_weights


# Training epoch
def train_epoch(trainloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing, device):
    encoder.train()
    decoder.train()
    total_loss = 0
    for train_input, train_target, input_len, target_len in trainloader:
        # train_input = torch.tensor(train_input)
        # train_target = torch.tensor(train_target)

        # Move tensor to GPU
        train_input = train_input.to(device)
        train_target = train_target.to(device)

        # Zero the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Forward pass
        encoder_outputs, encoder_hidden = encoder(train_input, input_len)
        decoder_outputs, decoder_hidden, attentions = decoder(train_input, encoder_outputs,encoder_hidden,train_target)

        # Mask of the padding part
        mask = (train_target != -1).float()
        
        # Compute loss
        #print(f"loss_computing:{decoder_outputs.shape,target_tensor.shape}")
        #loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)),train_target.view(-1, train_target.size(-1)))
        mse_loss = criterion(decoder_outputs, train_target)
        masked_loss = mse_loss*mask
        loss = masked_loss.sum()/mask.sum()

        # Backward pass and optimization
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(trainloader)

# Validation epoch
def eval_epoch(valloader, encoder, decoder, criterion, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    preds = []
    targets= []

    with torch.no_grad():
        for val_input, val_target, input_len, target_len in valloader:
            # to tensor
            # val_input = torch.tensor(val_input)
            # val_target = torch.tensor(val_target)

            # Move tensor to GPU
            val_input = val_input.to(device)
            val_target = val_target.to(device)

            # Forward pass
            encoder_outputs, encoder_hidden = encoder(val_input, input_len)
            decoder_outputs, _, _ = decoder(val_input, encoder_outputs,encoder_hidden)

            # Calculate loss
            mask = (val_target != -1).float()
            mse_loss = criterion(decoder_outputs, val_target)
            masked_loss = mse_loss*mask
            loss = masked_loss.sum()/mask.sum()
            
            total_loss += loss.item()

            pred = decoder_outputs.cpu().detach().numpy()
            target = val_target.cpu().detach().numpy()
            preds.append(pred)
            targets.append(target)
        
        # Calculate angle loss
        angle_actual = []
        angle_pred = []

        for target, pred in zip(targets, preds):
            angle_actual.append(target[:, :, -1])  # shape: [batch_size, seq_len]
            angle_pred.append(pred[:, :, -1])      # shape: [batch_size, seq_len]

        angle_actual_flatten = np.concatenate([seq.flatten() for seq in angle_actual])
        angle_pred_flatten = np.concatenate([seq.flatten() for seq in angle_pred])

        mse = mean_squared_error(angle_actual_flatten, angle_pred_flatten)
        mae = mean_absolute_error(angle_actual_flatten, angle_pred_flatten)

    return total_loss / len(valloader), mse, mae


# The overall training procedure
def running_train(train_dataloader, val_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, 
                    encoder_scheduler, decoder_scheduler, criterion, teacher_forcing, n_epochs, best_model_path, device):
    
    input_size = encoder.input_size
    hidden_size = encoder.hidden_size
    output_size = decoder.output_size

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    # dynamic disable tqdm output
    disable_tqdm = not sys.stdout.isatty()

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training Progress", disable=disable_tqdm):
        train_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing, device)
        train_losses.append(train_loss)
        
        val_loss, mse, mae = eval_epoch(val_dataloader, encoder, decoder, criterion, device)
        val_losses.append(val_loss)

        # Step the scheduler after each epoch
        encoder_scheduler.step()
        decoder_scheduler.step()

        #Save the model if the evaluation loss decreases
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch':epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'input_size': input_size,
                'output_size': output_size,
                'hidden_size': hidden_size,
                'loss':best_loss, #or val loss?
                'mse': mse,
                'mae': mae

            }, best_model_path)
            print(f"Model saved at epoch:{epoch}, mse:{mse}.", flush=True)

    return train_losses, val_losses

# The model should be called in your "training script".
# The following part is for unit test.
if __name__ == "__main__":
    # Set up some example parameters
    input_size = 7
    hidden_size = 128
    output_size = 7
    batch_size = 5
    seq_len = 20
    lr = 0.001
    epoch = 1
    
    test_inputs = torch.randn(batch_size, seq_len, input_size)
    target_tensor = torch.randn(batch_size, seq_len, output_size)
    dataset = TensorDataset(test_inputs, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #print(f"dataset shape: {test_inputs.shape,target_tensor.shape}")
    
    device = torch.device(f'cuda:1')
    best_model_path = "./test.pth"
    # Create Encoder, Decoder
    encoder = Encoder(input_size, hidden_size).to(device)
    decoder = AttnDecoder(hidden_size, output_size).to(device)

    # Create Optimizers
    encoder_opt = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    decoder_opt = optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)

    # Create LR schedulers
    encoder_scheduler = CosineAnnealingLR(encoder_opt, T_max=epoch)
    decoder_scheduler = CosineAnnealingLR(decoder_opt, T_max=epoch)

    # Define Loss
    criterion = nn.MSELoss()


    Loss = running_train(dataloader,dataloader,encoder,decoder,encoder_opt,decoder_opt,
                        encoder_scheduler,decoder_scheduler,criterion,teacher_forcing=0.5,
                        n_epochs=epoch,best_model_path=best_model_path,device=device)
    print(Loss)




