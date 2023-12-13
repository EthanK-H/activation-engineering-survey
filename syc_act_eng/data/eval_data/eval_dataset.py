import random
import json
import requests
from abc import ABC


'''
Dataset Template

'''

class DatasetTemplate(ABC):
    def __init__(self):
        '''
        Loads in the raw data
        '''
        pass

    def get_data_for_evaluation(self, user_tag='User: ', assistant_tag='Assistant: ', n_samples=100):
        '''
        Returns a list of evaluation examples.
        
        Each item in list is a dict with keys:
        - prompt (str): prompt text. 
        - answer_infos (dict[str: list]): dict any infos needed to do later evaluation
        '''
        
    def get_data_for_ninas_vector(self, n_samples=100):
        '''
        get data in a format ready to be passed to Nina's vector calculations
        
        Parameters:
        - n_samples (int): number of samples to return
        
        Returns:
            list[dict]: [
                {
                'sycophantic_text': str,
                'non_sycophantic_text': str
                }
            ]
        '''
        pass
        
    def get_data_for_reading_vector(self):
        raise NotImplementedError
    
    def evaluate_answers(
        self,
        model_answers: list[str: dict],
    ):
        '''
        Takes in model answers and eval infos and returns the evaluation results.
        
        model_answers is a list of dicts. Each dict has keys:
        - 'model_answer' (str): the model's answer
        - 'eval_info' (dict): any info needed to do the evaluation
        '''
        pass