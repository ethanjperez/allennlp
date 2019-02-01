import json
import logging
import os
from typing import Dict, List, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("race-mc")
class RaceReader(DatasetReader):
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
        self._using_bert = hasattr(self._token_indexers['tokens'], '_namespace') and self._token_indexers['tokens']._namespace == 'bert'
        if self._using_bert:
            print('BEEEEEEEEEEEEEEEEEEEERT!')

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    answer_texts = [answer['text'] for answer in question_answer['answers']]
                    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                    span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]
                    if self._using_bert:
                        # TODO: Add [SEP] cleanly with nlp.tokenizer.add_special_case('[SEP]', [{ORTH: '[SEP]', LEMMA: '[SEP]', POS: 'PUNCT'}]). Facilitates moving around Q/P/A
                        # Add Q to passage with a [SEP] token
                        char_question_span = (0, len(question_text))
                        tokenized_question = self._tokenizer.tokenize(question_text)
                        sep_str = '[SEP]'
                        prepend_text = question_text + ' ' + sep_str + ' '

                        # Adjust spans indices appropriately
                        span_starts = [len(prepend_text) + span_start for span_start in span_starts]
                        span_ends = [len(prepend_text) + span_end for span_end in span_ends]
                        char_answer_choice_spans = [(len(prepend_text) + char_answer_choice_span[0], len(prepend_text) + char_answer_choice_span[1])
                                                    for char_answer_choice_span in question_answer['char_answer_choice_spans']]

                        # Adjust passage token indices due to text added to passage
                        tokenized_question_paragraph = tokenized_question
                        tokenized_question_paragraph.append(Token(sep_str, len(question_text + ' ')))
                        for token in tokenized_paragraph:
                            new_token = Token(text=token.text, idx=token.idx+len(prepend_text), lemma=token.lemma,
                                              pos=token.pos, tag=token.tag, dep=token.dep, ent_type=token.ent_type)
                            tokenized_question_paragraph.append(new_token)
                        instance = self.text_to_instance(question_text,  # usually not used in this case but still given
                                                         prepend_text + paragraph,
                                                         zip(span_starts, span_ends),
                                                         answer_texts,
                                                         tokenized_question_paragraph,
                                                         question_answer['id'],
                                                         char_answer_choice_spans,
                                                         char_question_span)
                    else:
                        instance = self.text_to_instance(question_text,
                                                         paragraph,
                                                         zip(span_starts, span_ends),
                                                         answer_texts,
                                                         tokenized_paragraph,
                                                         question_answer['id'],
                                                         question_answer['char_answer_choice_spans'])
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None,
                         qa_id: str = None,
                         char_answer_choice_spans: List[Tuple[int, int]] = None,
                         char_question_span: Tuple[int, int] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                self.log_error(passage_text, passage_tokens, question_text, char_span_start, char_span_end, span_start, span_end)
            token_spans.append((span_start, span_end))

        additional_metadata = {'id': qa_id, 'answer_choice_spans': None, 'question_span': None}
        if char_answer_choice_spans is not None:
            additional_metadata['answer_choice_spans']: List[Tuple[int, int]] = []
            for char_answer_choice_span_start, char_answer_choice_span_end in char_answer_choice_spans:
                (answer_choice_span_start, answer_choice_span_end), error = util.char_span_to_token_span(passage_offsets, (char_answer_choice_span_start, char_answer_choice_span_end))
                if error:
                    self.log_error(passage_text, passage_tokens, question_text, char_answer_choice_span_start, char_answer_choice_span_end, answer_choice_span_start, answer_choice_span_end)
                additional_metadata['answer_choice_spans'].append((answer_choice_span_start, answer_choice_span_end))

        if char_question_span is not None:
            (question_span_start, question_span_end), error = util.char_span_to_token_span(passage_offsets, (char_question_span[0], char_question_span[1]))
            if error:
                self.log_error(passage_text, passage_tokens, question_text, char_question_span[0], char_question_span[1], question_span_start, question_span_end)
            additional_metadata['question_span'] = (question_span_start, question_span_end)

        return util.make_reading_comprehension_instance(self._tokenizer.tokenize(question_text),
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts,
                                                        additional_metadata)

    @staticmethod
    def log_error(passage_text, passage_tokens, question_text, char_span_start, char_span_end, span_start, span_end):
        """
        Logs an tokenization / span conversion error with the text info of the sample.
        """
        logger.debug("Passage: %s", passage_text)
        logger.debug("Passage tokens: %s", passage_tokens)
        logger.debug("Question text: %s", question_text)
        logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
        logger.debug("Token span: (%d, %d)", span_start, span_end)
        logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
        logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
        return


if __name__ == "__main__":
    # Parses RAW Race Files into Single JSON Format
    # Assumes race_raw directory lives in "datasets/race_raw" and you're running from allenlp dir
    race_raw_path = "datasets/race_raw"
    answer_tokens = ["1st", "2nd", "3rd", "4th"]
    letter_to_answer_token = {'A': "1st", 'B': "2nd", 'C': "3rd", 'D': "4th"}

    race_path = "datasets/race_mc"

    if not os.path.exists(race_path):
        os.mkdir(race_path)

    # Create Data Dictionary
    train_data, val_data, test_data = {"data": []}, {"data": []}, {"data": []}

    for dt in ['train', 'dev', 'test']:
        dt_path = os.path.join(race_raw_path, dt)
        for lvl in ['middle', 'high']:
            dt_lvl_path = os.path.join(dt_path, lvl)

            # Get all articles
            articles = os.listdir(dt_lvl_path)
            for article in articles:
                art_file = os.path.join(dt_lvl_path, article)
                with open(art_file, 'rb') as f:
                    art_data = json.load(f)

                # Set up top level json dict
                article_dict = {"title": art_data["id"], "paragraphs": []}

                # Iterate through questions
                for q in range(len(art_data["questions"])):
                    # Get base context
                    base_context = art_data["article"]

                    # Create Instance Dictionary
                    paragraph_dict = {}

                    # Get Question
                    question = art_data["questions"][q]

                    # Get Options
                    options = art_data["options"][q]

                    # Get Answer
                    answer = art_data["answers"][q]

                    # Build Context with all Options
                    answer_loc_to_predict, span_answer_text = None, None
                    pos_answers_text = ""
                    char_answer_choice_spans = []
                    for i, answer_token in enumerate(answer_tokens):
                        option_start_char = len(pos_answers_text)
                        pos_answers_text += options[i] + " "
                        if answer_token == letter_to_answer_token[answer]:
                            answer_loc_to_predict = len(pos_answers_text)
                            span_answer_text = answer_token
                        pos_answers_text += answer_token
                        option_end_char = len(pos_answers_text)
                        char_answer_choice_spans.append((option_start_char, option_end_char))
                        pos_answers_text += " "
                    base_context = pos_answers_text + base_context

                    # Get Q_ID
                    qid = hex(hash(art_file + question))[2:]

                    # Assemble dictionary
                    paragraph_dict["context"] = base_context
                    paragraph_dict["qas"] = [{"answers": [{"answer_start": answer_loc_to_predict,  # Official answer for prediction/evaluation is last token of the right multiple choice answer's actual text
                                                           "text": span_answer_text}],
                                              "question": question,
                                              "id": qid,
                                              "char_answer_choice_spans": char_answer_choice_spans}]  # Location of answer tokens (not directly predicted)

                    # Append to article dict
                    article_dict["paragraphs"].append(paragraph_dict)

                # Add article dict to data
                if dt == "train":
                    train_data["data"].append(article_dict)
                elif dt == "dev":
                    val_data["data"].append(article_dict)
                elif dt == "test":
                    test_data["data"].append(article_dict)

    # Dump JSON Files
    with open(os.path.join(race_path, "race-train-v1.0.json"), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(race_path, "race-dev-v1.0.json"), 'w') as f:
        json.dump(val_data, f)

    with open(os.path.join(race_path, "race-test-v1.0.json"), 'w') as f:
        json.dump(test_data, f)
