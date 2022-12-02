import torch
import argparse
import numpy as np
import random
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from omegaconf import OmegaConf
import torch.optim as optim
import data_loader.compdatasets as DataModule
import trainer as Trainer
import model.model as Model
import model.loss as Criterion
import model.metric as Metric
from wandb_setting import wandb_setting

def main(config):
    seed=2022
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    print('현재 적용되고 있는 모델은', config.model.model_name, '입니다.')
    
    # 데이터셋 로드 클래스를 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    
    # 데이터셋 로드 클래스를 불러옵니다.
    train = getattr(DataModule, config.model.data_class)(mode = "train", path = config.data.train_path, tokenizer=tokenizer, max_length=config.train.max_length)  
    valid = getattr(DataModule, config.model.data_class)(mode = "train", path = config.data.dev_path, tokenizer=tokenizer, max_length=config.train.max_length)
    
    train_dataloader = DataLoader(train, batch_size= config.train.batch_size, pin_memory=True, shuffle=True)
    valid_dataloader = DataLoader(valid, batch_size= config.train.batch_size, pin_memory=True, shuffle=False)

    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 아키텍처를 불러옵니다.
    print(f'현재 적용되고 있는 모델 클래스는 {config.model.model_class}입니다.')
    model = getattr(Model, config.model.model_class)(config.model.model_name, 1, config.model.dropout_rate).to(device)

    criterion = getattr(Criterion, config.model.loss)
    metric = getattr(Metric, config.model.metric)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    
    lr_scheduler = None
    epochs = config.train.max_epoch
    
    trainer = getattr(Trainer, config.model.trainer)(
            model = model,
            criterion = criterion,
            metric = metric,
            optimizer = optimizer,
            config = config,
            device = device,
            save_dir = config.model.saved_name,
            train_dataloader = train_dataloader,
            valid_dataloader = valid_dataloader,
            lr_scheduler=lr_scheduler,
            epochs=epochs,
            tokenizer = tokenizer
        )
    
    trainer.train()

if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()
    
    config_w = OmegaConf.load(f'./configs/{args.config}.yaml')
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')

    main(config_w)