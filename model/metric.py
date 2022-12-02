import numpy as np
import einops as ein

def pearson_correlation(output, target):
    '''
    input (Tensor) - A 2D matrix containing multiple variables and observations, or a Scalar or 1D vector representing a single variable
    입력값(텐서) = 변수들과 관찰값들을 포함하는 2차원 행렬 또는 하나의 변수를 의미하는 1차원 벡터 또는 스칼라
    '''
    output = output.flatten()
    target = target.flatten()

    return np.corrcoef(output, target)[0, 1]