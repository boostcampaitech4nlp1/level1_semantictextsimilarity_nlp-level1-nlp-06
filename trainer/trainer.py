from tqdm import tqdm
import torch
import numpy as np
import os
import gc
import time
import re

class Trainer():
    """
    훈련과정입니다.
    """
    def __init__(self, model, criterion, metric, optimizer, config, device, save_dir,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None, epochs=1, tokenizer=None):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.tokenizer = tokenizer

    def train(self):
        """
        train_epoch를 돌고 valid_epoch로 평가합니다.
        """
        for epoch in range(self.epochs):
            standard_time = time.time()
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
        torch.cuda.empty_cache()
        del self.model, self.train_dataloader, self.valid_dataloader
        gc.collect()
    
    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        epoch_loss = 0
        steps = 0
        pbar = tqdm(self.train_dataloader)
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            steps += 1
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            label = batch["labels"].to(self.device)
            logits = self.model(input_ids, attention_mask)
            
            loss = self.criterion(logits, label)    
            loss.backward()
            epoch_loss += loss.detach().cpu().numpy().item()
            
            self.optimizer.step()
            
            pbar.set_postfix({
                'loss' : epoch_loss / steps,
                'lr' : self.optimizer.param_groups[0]['lr'],
            })

        pbar.close()

    def _valid_epoch(self, epoch):
        val_loss = 0
        val_steps = 0
        total_val_score=0
        val_loss_values=[1]
        with torch.no_grad():
            self.model.eval()
            for valid_batch in tqdm(self.valid_dataloader):
                input_ids = valid_batch["input_ids"].to(self.device)
                attention_mask = valid_batch["attention_mask"].to(self.device)
                label = valid_batch["labels"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)

                val_steps += 1
                
                loss = self.criterion(logits, label)
                val_loss += loss.detach().cpu().numpy().item()                
                total_val_score += self.metric(logits, label).item()
            
            val_loss /= val_steps
            total_val_score /= val_steps
            print(f"Epoch [{epoch+1}/{self.epochs}] Val_loss : {val_loss}")
            print(f"Epoch [{epoch+1}/{self.epochs}] Score : {total_val_score}")

            if min(val_loss_values) >= val_loss and val_loss<0.5:
                print('save checkpoint!')
                if not os.path.exists(f'save/{self.save_dir}'):
                    os.makedirs(f'save/{self.save_dir}')
                torch.save(self.model.state_dict(), f'save/{self.save_dir}/epoch:{epoch}_model.pt')
                val_loss_values.append(val_loss)
            
class T5Trainer():
    """
    T5 트레이너
    """
    def __init__(self, model, criterion, metric, optimizer, config, device, save_dir,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None, epochs=1, tokenizer=None):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.tokenizer = tokenizer

    def train(self):
        """
        train_epoch를 돌고 valid_epoch로 평가합니다.
        """
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
        torch.cuda.empty_cache()
        del self.model, self.train_dataloader, self.valid_dataloader
        gc.collect()
    
    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        epoch_loss = 0
        steps = 0
        pbar = tqdm(self.train_dataloader)
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            steps += 1
            
            input_ids = batch["src_input_ids"].to(self.device)
            attention_mask = batch["src_attention_mask"].to(self.device)
            label = batch["tgt_input_ids"].to(self.device)
            outputs = self.model(input_ids, attention_mask, label)
            
            loss = outputs.loss
            loss.backward()
            
            epoch_loss += loss.detach().cpu().numpy().item()
            
            self.optimizer.step()
            
            pbar.set_postfix({
                'loss' : epoch_loss / steps,
                'lr' : self.optimizer.param_groups[0]['lr'],
            })

        pbar.close()

    def _valid_epoch(self, epoch):
        val_loss = 0
        val_steps = 0
        total_val_score=0
        val_loss_values=[1]
        with torch.no_grad():
            self.model.eval()
            all_preds, all_labels = [], []
            for valid_batch in tqdm(self.valid_dataloader):
                val_steps += 1
                
                input_ids = valid_batch["src_input_ids"].to(self.device)
                attention_mask = valid_batch["src_attention_mask"].to(self.device)
                label = valid_batch["tgt_input_ids"].to(self.device)
                
                for true_id in label:
                    true_decoded = self.tokenizer.decode(true_id)
                    all_labels.append(float(true_decoded.split('score:')[1].split('</s>')[0]))
                
                outputs = self.model(input_ids, attention_mask, label)
                
                loss = outputs.loss
                val_loss += loss.detach().cpu().numpy().item()
                pred_ids = self.model.model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
                pred_ids = pred_ids.detach().cpu().numpy()
                for pred_id in pred_ids:
                    pred_decoded = self.tokenizer.decode(pred_id)
                    all_preds.append(float(pred_decoded.split('score:')[1].split('</s>')[0]))
            
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            total_val_score += self.metric(all_labels, all_preds).item()
            
            val_loss /= val_steps
            print(f"Epoch [{epoch+1}/{self.epochs}] Score : {total_val_score}")
            print(f"Epoch [{epoch+1}/{self.epochs}] Val_loss : {val_loss}")

            if min(val_loss_values) >= val_loss and val_loss<0.5:
                print('save checkpoint!')
                if not os.path.exists(f'save/{self.save_dir}'):
                    os.makedirs(f'save/{self.save_dir}')
                torch.save(self.model.state_dict(), f'save/{self.save_dir}/epoch:{epoch}_model.pt')
                val_loss_values.append(val_loss)