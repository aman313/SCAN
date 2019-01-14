from typing import Dict
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from random import shuffle
import random
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("seq2seq_scan_pair")
class Seq2SeqSCANPairDatasetReader(DatasetReader):
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 same_inst_batch: bool=False,
                 same_inst_batch_size: int =-1,
                 ref_path = None,
                 max_ref = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        #self._ref_path = ref_path
        self._ref_path = 'add_prim_split/tasks_train_devremoved_addprim_jump.txt'
        self._max_ref = max_ref
        self._same_inst_batch = same_inst_batch
        if same_inst_batch and not same_inst_batch_size > 0:
            raise Exception('if same_inst_batch, batch size should be positive')
        self._same_inst_batch_size = same_inst_batch_size

    @overrides
    def _read(self, file_path):
        if self._ref_path == None:
            self._ref_path = file_path
        references = []
        with open(self._ref_path) as rf:
            for line in rf:
                inp = line.split('OUT:')[0].split('IN:')[1].strip()
                out = line.split('OUT:')[1].strip()
                #if 'jump' not in inp.lower():
                #    continue
                references.append((inp,out))
        shuffle(references)

        if self._max_ref>0:
            references = references[:self._max_ref] 
        
        with open(file_path) as fl:
            for line in fl:
                inp = line.split('OUT:')[0].split('IN:')[1].strip()
                out = line.split('OUT:')[1].strip()
                if self._same_inst_batch:
                    for i in range(self._same_inst_batch_size):
                        random_ref = random.choice(references)
                        yield self.text_to_instance(inp, random_ref[0],random_ref[1],out)
                else:
                    random_ref = random.choice(references)
                    yield self.text_to_instance(inp, random_ref[0],random_ref[1],out)
                    

    @overrides
    def text_to_instance(self, source_string: str,reference_source_string:str,reference_target_string:str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_ref_source = self._source_tokenizer.tokenize(reference_source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
            tokenized_ref_source.insert(0,Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        tokenized_ref_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        reference_source_field = TextField(tokenized_ref_source,self._source_token_indexers)
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            
            tokenized_ref_target = self._target_tokenizer.tokenize(reference_target_string)
            tokenized_ref_target.insert(0, Token(START_SYMBOL))
            tokenized_ref_target.append(Token(END_SYMBOL))
            ref_target_field = TextField(tokenized_ref_target, self._target_token_indexers)

            return Instance({"instance_source_tokens": source_field, "instance_target_tokens": target_field,"ref_source_tokens":reference_source_field,"ref_target_tokens":ref_target_field})
        else:
            tokenized_ref_target = self._target_tokenizer.tokenize(reference_target_string)
            tokenized_ref_target.insert(0, Token(START_SYMBOL))
            tokenized_ref_target.append(Token(END_SYMBOL))
            ref_target_field = TextField(tokenized_ref_target, self._target_token_indexers)

            return Instance({"instance_source_tokens": source_field,"ref_source_tokens":reference_source_field,"ref_target_tokens":ref_target_field})
