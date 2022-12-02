import numpy
from typing import Callable
import pandas as pd
import einops as ein
import torch

class CompDataset(torch.utils.data.Dataset):
    """_summary_
    데이터를 불러와 전처리와 토크나이저 등 다양한 전처리를 수행하고
    data와 target을 나눠주는 작업을 해주는 클래스입니다.
    """
    def __init__(self, 
                 mode: str, # train / test 모드를 설정해줍니다.
                 path: str, # 데이터셋을 불러올 root path를 지정해줍니다.
                 tokenizer: Callable,
                 max_length: int = 512, # 토크나이징할 문장의 최대 길이를 설정해줍니다.
                 ):
        super().__init__()
        self.mode = mode
        self.max_length = max_length
        
        if self.mode:
            self.sentence_1_array, self.sentence_2_array, self.target_array = self._load_data(path)
        else:
            self.sentence_1_array, self.sentence_2_array = self._load_data(path)
        
        self.tokenizer = tokenizer

    def _load_data(self, path:str) -> numpy.ndarray:
        """_summary_
        데이터 컬럼 : id, source, sentence_1, sentence_2, label, binary_label
        그 중에 필요한 컬럼 : [features](str) : sentence_1, sentence_2
                              [target](float) : label
        Returns:
            sentence_1(str) : 비교할 첫번째 문장
            sentence_2(str) : 비교할 두번째 문장
            target(Optional[float])
        """
        # root path 안의 mode에 해당하는 csv 파일을 가져옵니다.
        df = pd.read_csv(path)
        sentence_1 = df['sentence_1'].to_numpy()
        sentence_2 = df['sentence_2'].to_numpy()
        if self.mode: # train or validation일 경우
            target = df['label'].to_numpy()
            
            return sentence_1, sentence_2, target
        else: # test일 경우
            return sentence_1, sentence_2
    
    def _preprocess(self, x:str) -> str:
        # 데이터 전처리를 위한 함수입니다.
        
        return x

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self) -> int:
        return len(self.sentence_1_array)
    
    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 토크나이저 및 전처리를 위해 두 문장을 하나로 합쳐줍니다.
        sentence_1 = self._preprocess(self.sentence_1_array[idx])
        sentence_2 = self._preprocess(self.sentence_2_array[idx])
        
        encoded_dict = self.tokenizer.encode_plus(
            sentence_1,
            sentence_2,           
            add_special_tokens = True,      
            max_length = self.max_length,           
            pad_to_max_length = True, # 여기서 이미 패딩을 수행합니다.
            truncation=True,
            return_attention_mask = True,   
            return_tensors = 'pt',
            )
        
        if self.mode:
            return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s'), 
                    'labels': ein.rearrange(torch.tensor(self.target_array[idx], dtype=torch.float), ' -> 1')}
        else:
            return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s')}
            
class T5encoderDataset(torch.utils.data.Dataset):
    """_summary_
    데이터를 불러와 전처리와 토크나이저 등 다양한 전처리를 수행하고
    data와 target을 나눠주는 작업을 해주는 클래스입니다.
    """
    def __init__(self, 
                 mode: str, # train / test 모드를 설정해줍니다.
                 path: str, # 데이터셋을 불러올 root path를 지정해줍니다.
                 tokenizer: Callable,
                 max_length: int = 512, # 토크나이징할 문장의 최대 길이를 설정해줍니다.
                 ):
        super().__init__()
        self.mode = mode
        self.max_length = max_length
        
        if self.mode:
            self.sentence_1_array, self.sentence_2_array, self.target_array = self._load_data(path)
        else:
            self.sentence_1_array, self.sentence_2_array = self._load_data(path)
        
        self.tokenizer = tokenizer

    def _load_data(self, path:str) -> numpy.ndarray:
        """_summary_
        데이터 컬럼 : id, source, sentence_1, sentence_2, label, binary_label
        그 중에 필요한 컬럼 : [features](str) : sentence_1, sentence_2
                              [target](float) : label
        Returns:
            sentence_1(str) : 비교할 첫번째 문장
            sentence_2(str) : 비교할 두번째 문장
            target(Optional[float])
        """
        # root path 안의 mode에 해당하는 csv 파일을 가져옵니다.
        df = pd.read_csv(path)
        sentence_1 = df['sentence_1'].to_numpy()
        sentence_2 = df['sentence_2'].to_numpy()
        if self.mode: # train or validation일 경우
            target = df['label'].to_numpy()
            
            return sentence_1, sentence_2, target
        else: # test일 경우
            return sentence_1, sentence_2
    
    def _preprocess(self, x:str) -> str:
        # 데이터 전처리를 위한 함수입니다.
        
        return x

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self) -> int:
        return len(self.sentence_1_array)
    
    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 토크나이저 및 전처리를 위해 두 문장을 하나로 합쳐줍니다.
        text = f'korsts sentence1: {self._preprocess(self.sentence_1_array[idx])} sentence2: {self._preprocess(self.sentence_2_array[idx])}'
        
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,      
            max_length = self.max_length,           
            pad_to_max_length = True, # 여기서 이미 패딩을 수행합니다.
            truncation=True,
            return_attention_mask = True,   
            return_tensors = 'pt',
            )
        
        if self.mode:
            return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s'), 
                    'labels': ein.rearrange(torch.tensor(self.target_array[idx], dtype=torch.float), ' -> 1')}
        else:
            return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s')}
            
class T5Dataset(torch.utils.data.Dataset):
    """_summary_
    데이터를 불러와 전처리와 토크나이저 등 다양한 전처리를 수행하고
    data와 target을 나눠주는 작업을 해주는 클래스입니다.
    """
    def __init__(self, 
                 mode: str, # train / test 모드를 설정해줍니다.
                 path: str, # 데이터셋을 불러올 root path를 지정해줍니다.
                 tokenizer: Callable,
                 max_length: int = 512, # 토크나이징할 문장의 최대 길이를 설정해줍니다.
                 ):
        super().__init__()
        self.mode = mode
        self.max_length = max_length
        
        if self.mode:
            self.sentence_1_array, self.sentence_2_array, self.target_array = self._load_data(path)
        else:
            self.sentence_1_array, self.sentence_2_array = self._load_data(path)
        
        self.tokenizer = tokenizer

    def _load_data(self, path:str) -> numpy.ndarray:
        """_summary_
        데이터 컬럼 : id, source, sentence_1, sentence_2, label, binary_label
        그 중에 필요한 컬럼 : [features](str) : sentence_1, sentence_2
                              [target](float) : label
        Returns:
            sentence_1(str) : 비교할 첫번째 문장
            sentence_2(str) : 비교할 두번째 문장
            target(Optional[float])
        """
        # root path 안의 mode에 해당하는 csv 파일을 가져옵니다.
        df = pd.read_csv(path)
        sentence_1 = df['sentence_1'].to_numpy()
        sentence_2 = df['sentence_2'].to_numpy()
        if self.mode: # train or validation일 경우
            target = df['label'].to_numpy()
            
            return sentence_1, sentence_2, target
        else: # test일 경우
            return sentence_1, sentence_2
    
    def _preprocess(self, x:str) -> str:
        # 데이터 전처리를 위한 함수입니다.
        
        return x

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self) -> int:
        return len(self.sentence_1_array)
    
    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 토크나이저 및 전처리를 위해 두 문장을 하나로 합쳐줍니다.
        sentence_1 = self._preprocess(self.sentence_1_array[idx])
        sentence_2 = self._preprocess(self.sentence_2_array[idx])
        
        text = f'korsts sentence1: {sentence_1} sentence2: {sentence_2}'
        encoded_dict = self.tokenizer.encode_plus(
            text,   
            add_special_tokens = True,      
            max_length = self.max_length,           
            pad_to_max_length = True,
            truncation=True,
            return_attention_mask = True,   
            return_tensors = 'pt',          
            )
        src_input_ids = ein.rearrange(encoded_dict['input_ids'], '1 s -> s')
        src_att_mask = ein.rearrange(encoded_dict['attention_mask'], '1 s -> s')
        
        if self.mode: #train, val
            label = f'score:{self.target_array[idx]}'

            target_dict = self.tokenizer.encode_plus(
                label,           
                add_special_tokens = True,      
                max_length = self.max_length,           
                pad_to_max_length = True,
                truncation=True,
                return_attention_mask = True,   
                return_tensors = 'pt',          
                )
            tgt_input_ids = ein.rearrange(target_dict['input_ids'], '1 s -> s')
            tgt_attention_mask = ein.rearrange(target_dict['attention_mask'], '1 s -> s')
            
            return {'src_input_ids': src_input_ids.long(), 
                    'src_attention_mask': src_att_mask.long(), 
                    'tgt_input_ids': tgt_input_ids.long(),
                    }
        else: # test
            return {
                    'src_input_ids': src_input_ids.long(),
                    'src_attention_mask': src_att_mask.long()
                    }