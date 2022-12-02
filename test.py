import torch
import argparse
from tqdm import tqdm
import einops as ein
import numpy as np
import pandas as pd
import random

from transformers import AutoTokenizer
from omegaconf import OmegaConf
import data_loader.compdatasets as DataModule
import model.model as Model
from wandb_setting import wandb_setting

def main(config):
    random_seed=2022
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    # 데이터셋 로드 클래스를 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    test_dataloader = getattr(DataModule, config.model.data_class)(mode=False,
                                                data = 'test',
                                                model_name=config.model.model_name,
                                                max_length=config.train.max_length,
                                                batch_size = 1,
                                                shuffle=False)

    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 아키텍처를 불러옵니다.
    model = getattr(Model, config.model.model_class)(config.model.model_name, 1, config.model.dropout_rate).to(device)
    checkpoint = torch.load(f'./save/epoch:{config.train.max_epoch-1}_model.pt')
    model.load_state_dict(checkpoint)
    
    # criterion = getattr(Criterion, config.model.loss)
    # metric = getattr(Metric, config.model.metric)
    # optimizer = getattr(Optimizer, config.model.optimizer)
    
    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    all_preds = []
    with torch.no_grad():
        if 't5' in config.model.model_name:
            for batch in tqdm(test_dataloader):
                input_ids = batch["src_input_ids"].to(device)
                attention_mask = batch["src_attention_mask"].to(device)
                pred_ids = model.model.generate(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                pred_ids = pred_ids.squeeze().detach().cpu().numpy()
                pred_decoded = tokenizer.decode(pred_ids)
                preds = float(pred_decoded.split('Score -\t')[1].split('</s>')[0])
                preds = round(preds, 1)
                all_preds.append(preds)
        else:
            for batch in tqdm(test_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                output = model(input_ids, attention_mask)
                preds = ein.rearrange(output, 'b 1 -> b').detach().cpu().numpy().item()
                preds = np.round(preds, 1)
                all_preds.append(preds)
    test_df = pd.read_csv('data/' + 'test.csv')
    test_df['target']=all_preds
    test_df[['id', 'target']].to_csv('save/submission.csv')
    print('추론 완료')

if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()

    config_w = OmegaConf.load(f'./configs/{args.config}.yaml')
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')
    
    main(config_w)
