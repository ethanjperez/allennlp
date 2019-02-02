import json
import logging
import os
from typing import Dict, List

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, IndexField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("race-mc")
class RaceMCReader(DatasetReader):
    """
    Reads a JSON-formatted Race file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._letter_to_answer_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading files at %s", file_path)
        for level in ['middle', 'high']:
            # Get all articles
            file_level_path = os.path.join(file_path, level)
            articles = os.listdir(file_level_path)
            for article in articles:
                art_file = os.path.join(file_level_path, article)
                with open(art_file, 'rb') as f:
                    art_data = json.load(f)

                # Article-level info
                title = art_data["id"]
                passage_text = art_data["article"]
                tokenized_passage = self._tokenizer.tokenize(passage_text)

                # Iterate through questions
                for q in range(len(art_data["questions"])):
                    question_text = art_data["questions"][q].strip().replace("\n", "")
                    options_text = art_data["options"][q]
                    answer_index = self._letter_to_answer_idx[art_data["answers"][q]]
                    qid = os.path.join(art_file, str(q))
                    yield self.text_to_instance(question_text,
                                                passage_text,
                                                options_text,
                                                qid,
                                                tokenized_passage,
                                                answer_index)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         options_text: List[str],
                         qa_id: str,
                         passage_tokens: List[Token] = None,
                         answer_index: int = None) -> Instance:
        # pylint: disable=arguments-differ
        additional_metadata = {'id': qa_id}
        fields: Dict[str, Field] = {}

        # Tokenize and calculate token offsets (i.e., for wordpiece)
        question_tokens = self._tokenizer.tokenize(question_text)
        if passage_tokens is None:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        options_tokens = self._tokenizer.batch_tokenize(options_text)
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

        # This is separate so we can reference it later with a known type.
        options_field = ListField([TextField(option_tokens, self._token_indexers) for option_tokens in options_tokens])
        fields['passage'] = TextField(passage_tokens, self._token_indexers)
        fields['question'] = TextField(question_tokens, self._token_indexers)
        fields['options'] = options_field
        metadata = {'original_passage': passage_text, 'token_offsets': passage_offsets,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [token.text for token in passage_tokens],
                    'options_tokens': [[token.text for token in option_tokens] for option_tokens in options_tokens]}
        if answer_index is not None:
            metadata['answer_texts'] = options_text[answer_index]

        fields['answer_index'] = IndexField(answer_index, options_field)

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)

# if __name__ == "__main__":
#     # Parses RAW Race Files into Single JSON Format
#     # Assumes race_raw directory lives in "datasets/race_raw" and you're running from allenlp dir
#     race_raw_path = "datasets/race_raw"
#     race_path = "datasets/race_mc"
#     splits = ['train', 'dev', 'test']
#     letter_to_answer_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
#
#     if not os.path.exists(race_path):
#         os.mkdir(race_path)
#
#     # Create Data Dictionary
#     data = {}
#     for split in splits:
#         data[split] = {'data': []}
#         split_path = os.path.join(race_raw_path, split)
#         for level in ['middle', 'high']:
#             split_level_path = os.path.join(split_path, level)
#
#             # Get all articles
#             articles = os.listdir(split_level_path)
#             for article in articles:
#                 art_file = os.path.join(split_level_path, article)
#                 with open(art_file, 'rb') as f:
#                     art_data = json.load(f)
#
#                 # Set up top level json dict
#                 article_dict = {"title": art_data["id"], "paragraphs": []}
#
#                 # Iterate through questions
#                 for q in range(len(art_data["questions"])):
#                     # Create Instance Dictionary
#                     article_dict["paragraphs"].append({
#                         'context': art_data["article"],
#                         'qas': [{
#                             "options": art_data["options"][q],
#                             "answers": [letter_to_answer_idx[art_data["answers"][q]]],
#                             "question": art_data["questions"][q],
#                             "id": art_file
#                         }]
#                     })
#
#                 # Add article dict to data
#                 data[split]['data'].append(article_dict)
#
#     # Dump JSON Files
#     for split in splits:
#         with open(os.path.join(race_path, "race-" + split + "-v1.0.json"), 'w') as f:
#             json.dump(data[split], f)
