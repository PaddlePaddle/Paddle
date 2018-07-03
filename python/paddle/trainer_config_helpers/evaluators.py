# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.trainer.config_parser import *
from default_decorators import *

__all__ = [
    "evaluator_base",
    "classification_error_evaluator",
    "auc_evaluator",
    "pnpair_evaluator",
    "precision_recall_evaluator",
    "ctc_error_evaluator",
    "chunk_evaluator",
    "sum_evaluator",
    "column_sum_evaluator",
    "value_printer_evaluator",
    "gradient_printer_evaluator",
    "maxid_printer_evaluator",
    "maxframe_printer_evaluator",
    "seqtext_printer_evaluator",
    "classification_error_printer_evaluator",
    "detection_map_evaluator",
]


class EvaluatorAttribute(object):
    FOR_CLASSIFICATION = 1
    FOR_REGRESSION = 1 << 1
    FOR_RANK = 1 << 2
    FOR_PRINT = 1 << 3
    FOR_UTILS = 1 << 4
    FOR_DETECTION = 1 << 5

    KEYS = [
        "for_classification", "for_regression", "for_rank", "for_print",
        "for_utils", "for_detection"
    ]

    @staticmethod
    def to_key(idx):
        tmp = 1
        for i in xrange(0, len(EvaluatorAttribute.KEYS)):
            if idx == tmp:
                return EvaluatorAttribute.KEYS[i]
            else:
                tmp = (tmp << 1)


def evaluator(*attrs):
    def impl(method):
        for attr in attrs:
            setattr(method, EvaluatorAttribute.to_key(attr), True)
        method.is_evaluator = True
        return method

    return impl


def evaluator_base(input,
                   type,
                   label=None,
                   weight=None,
                   name=None,
                   chunk_scheme=None,
                   num_chunk_types=None,
                   classification_threshold=None,
                   positive_label=None,
                   dict_file=None,
                   result_file=None,
                   num_results=None,
                   delimited=None,
                   top_k=None,
                   excluded_chunk_types=None,
                   overlap_threshold=None,
                   background_id=None,
                   evaluate_difficult=None,
                   ap_type=None):
    """
    Evaluator will evaluate the network status while training/testing.

    User can use evaluator by classify/regression job. For example.

    ..  code-block:: python

        classify(prediction, output, evaluator=classification_error_evaluator)

    And user could define evaluator separately as follow.

    ..  code-block:: python

        classification_error_evaluator("ErrorRate", prediction, label)

    The evaluator often contains a name parameter. It will also be printed when
    evaluating network. The printed information may look like the following.

    ..  code-block:: text

         Batch=200 samples=20000 AvgCost=0.679655 CurrentCost=0.662179 Eval:
         classification_error_evaluator=0.4486
         CurrentEval: ErrorRate=0.3964

    :param input: Input layers, a object of LayerOutput or a list of
                  LayerOutput.
    :type input: list|LayerOutput
    :param label: An input layer containing the ground truth label.
    :type label: LayerOutput|None
    :param weight: An input layer which is a weight for each sample.
                   Each evaluator may calculate differently to use this weight.
    :type weight: LayerOutput.
    :param top_k: number k in top-k error rate
    :type top_k: int
    :param overlap_threshold: In detection tasks to filter detection results
    :type overlap_threshold: float
    :param background_id: Identifier of background class
    :type background_id: int
    :param evaluate_difficult: Whether to evaluate difficult objects
    :type evaluate_difficult: bool
    :param ap_type: How to calculate average persicion
    :type ap_type: str
    """
    # inputs type assertions.
    assert classification_threshold is None or isinstance(
        classification_threshold, float)
    assert positive_label is None or isinstance(positive_label, int)
    assert num_results is None or isinstance(num_results, int)
    assert top_k is None or isinstance(top_k, int)

    if not isinstance(input, list):
        input = [input]

    if label:
        input.append(label)
    if weight:
        input.append(weight)

    Evaluator(
        name=name,
        type=type,
        inputs=[i.name for i in input],
        chunk_scheme=chunk_scheme,
        num_chunk_types=num_chunk_types,
        classification_threshold=classification_threshold,
        positive_label=positive_label,
        dict_file=dict_file,
        result_file=result_file,
        delimited=delimited,
        num_results=num_results,
        top_k=top_k,
        excluded_chunk_types=excluded_chunk_types,
        overlap_threshold=overlap_threshold,
        background_id=background_id,
        evaluate_difficult=evaluate_difficult,
        ap_type=ap_type)


@evaluator(EvaluatorAttribute.FOR_DETECTION)
@wrap_name_default()
def detection_map_evaluator(input,
                            label,
                            overlap_threshold=0.5,
                            background_id=0,
                            evaluate_difficult=False,
                            ap_type="11point",
                            name=None):
    """
    Detection mAP Evaluator. It will print mean Average Precision (mAP) for detection.

    The detection mAP Evaluator based on the output of detection_output layer counts
    the true positive and the false positive bbox and integral them to get the
    mAP.

    The simple usage is:

    .. code-block:: python

       eval =  detection_map_evaluator(input=det_output,label=lbl)

    :param input: Input layer.
    :type input: LayerOutput
    :param label: Label layer.
    :type label: LayerOutput
    :param overlap_threshold: The bbox overlap threshold of a true positive.
    :type overlap_threshold: float
    :param background_id: The background class index.
    :type background_id: int
    :param evaluate_difficult: Whether evaluate a difficult ground truth.
    :type evaluate_difficult: bool
    """
    if not isinstance(input, list):
        input = [input]

    if label:
        input.append(label)

    evaluator_base(
        name=name,
        type="detection_map",
        input=input,
        label=label,
        overlap_threshold=overlap_threshold,
        background_id=background_id,
        evaluate_difficult=evaluate_difficult,
        ap_type=ap_type)


@evaluator(EvaluatorAttribute.FOR_CLASSIFICATION)
@wrap_name_default()
def classification_error_evaluator(input,
                                   label,
                                   name=None,
                                   weight=None,
                                   top_k=None,
                                   threshold=None):
    """
    Classification Error Evaluator. It will print error rate for classification.

    The classification error is:

    ..  math::

        classification\\_error = \\frac{NumOfWrongPredicts}{NumOfAllSamples}

    The simple usage is:

    .. code-block:: python

       eval =  classification_error_evaluator(input=prob,label=lbl)

    :param name: Evaluator name.
    :type name: basestring
    :param input: Input Layer name. The output prediction of network.
    :type input: LayerOutput
    :param label: Label layer name.
    :type label: basestring
    :param weight: Weight Layer name. It should be a matrix with size
                  [sample_num, 1]. And will just multiply to NumOfWrongPredicts
                  and NumOfAllSamples. So, the elements of weight are all one,
                  then means not set weight. The larger weight it is, the more
                  important this sample is.
    :type weight: LayerOutput
    :param top_k: number k in top-k error rate
    :type top_k: int
    :param threshold: The classification threshold.
    :type threshold: float
    :return: None.
    """

    evaluator_base(
        name=name,
        type="classification_error",
        input=input,
        label=label,
        weight=weight,
        top_k=top_k,
        classification_threshold=threshold, )


@evaluator(EvaluatorAttribute.FOR_CLASSIFICATION)
@wrap_name_default()
def auc_evaluator(
        input,
        label,
        name=None,
        weight=None, ):
    """
    Auc Evaluator which adapts to binary classification.

    The simple usage:

    .. code-block:: python

       eval = auc_evaluator(input, label)

    :param name: Evaluator name.
    :type name: None|basestring
    :param input: Input Layer name. The output prediction of network.
    :type input: LayerOutput
    :param label: Label layer name.
    :type label: None|basestring
    :param weight: Weight Layer name. It should be a matrix with size
                  [sample_num, 1].
    :type weight: LayerOutput
    """
    evaluator_base(
        name=name,
        type="last-column-auc",
        input=input,
        label=label,
        weight=weight)


@evaluator(EvaluatorAttribute.FOR_RANK)
@wrap_name_default()
def pnpair_evaluator(
        input,
        label,
        query_id,
        weight=None,
        name=None, ):
    """
    Positive-negative pair rate Evaluator which adapts to rank task like
    learning to rank. This evaluator must contain at least three layers.

    The simple usage:

    .. code-block:: python

       eval = pnpair_evaluator(input, label, query_id)

    :param input: Input Layer name. The output prediction of network.
    :type input: LayerOutput
    :param label: Label layer name.
    :type label: LayerOutput
    :param query_id: Query_id layer name. Query_id indicates that which query
     each sample belongs to. Its shape should be
     the same as output of Label layer.
    :type query_id: LayerOutput
    :param weight: Weight Layer name. It should be a matrix with size
                  [sample_num, 1] which indicates the weight of each sample.
                  The default weight of sample is 1 if the weight layer is None.
                  And the pair weight is the mean of the two samples' weight.
    :type weight: LayerOutput
    :param name: Evaluator name.
    :type name: None|basestring
    """
    if not isinstance(input, list):
        input = [input]
    if label:
        input.append(label)
    if query_id:
        input.append(query_id)
    evaluator_base(
        input=input,
        type="pnpair",
        weight=weight,
        name=name, )


@evaluator(EvaluatorAttribute.FOR_CLASSIFICATION)
@wrap_name_default()
def precision_recall_evaluator(
        input,
        label,
        positive_label=None,
        weight=None,
        name=None, ):
    """
    An Evaluator to calculate precision and recall, F1-score.
    It is adapt to the task with multiple labels.

    - If positive_label=-1, it will print the average precision, recall,
      F1-score of all labels.

    - If use specify positive_label, it will print the precision, recall,
      F1-score of this label.

    The simple usage:

    .. code-block:: python

       eval = precision_recall_evaluator(input, label)

    :param name: Evaluator name.
    :type name: None|basestring
    :param input: Input Layer name. The output prediction of network.
    :type input: LayerOutput
    :param label: Label layer name.
    :type label: LayerOutput
    :param positive_label: The input label layer.
    :type positive_label: LayerOutput.
    :param weight: Weight Layer name. It should be a matrix with size
                  [sample_num, 1]. (TODO, explaination)
    :type weight: LayerOutput
    """
    evaluator_base(
        name=name,
        type="precision_recall",
        input=input,
        label=label,
        positive_label=positive_label,
        weight=weight)


@evaluator(EvaluatorAttribute.FOR_CLASSIFICATION)
@wrap_name_default()
def ctc_error_evaluator(
        input,
        label,
        name=None, ):
    """
    This evaluator is to calculate sequence-to-sequence edit distance.

    The simple usage is :

    .. code-block:: python

       eval = ctc_error_evaluator(input=input, label=lbl)

    :param name: Evaluator name.
    :type name: None|basestring
    :param input: Input Layer. Should be the same as the input for ctc_layer.
    :type input: LayerOutput
    :param label: input label, which is a data_layer. Should be the same as the
                  label for ctc_layer
    :type label: LayerOutput
    """
    evaluator_base(
        name=name, type="ctc_edit_distance", input=input, label=label)


@evaluator(EvaluatorAttribute.FOR_CLASSIFICATION)
@wrap_name_default()
def chunk_evaluator(
        input,
        label,
        chunk_scheme,
        num_chunk_types,
        name=None,
        excluded_chunk_types=None, ):
    """
    Chunk evaluator is used to evaluate segment labelling accuracy for a
    sequence. It calculates precision, recall and F1 scores for the chunk detection.

    To use chunk evaluator, several concepts need to be clarified firstly.

    * **Chunk type** is the type of the whole chunk and a chunk consists of one or several words.  (For example in NER, ORG for organization name, PER for person name etc.)

    * **Tag type** indicates the position of a word in a chunk. (B for begin, I for inside, E for end, S for single)
    We can name a label by combining tag type and chunk type. (ie. B-ORG for begining of an organization name)

    The construction of label dictionary should obey the following rules:

    - Use one of the listed labelling schemes. These schemes differ in ways indicating chunk boundry.

    .. code-block:: text

        Scheme    Description
        plain    Use the same label for the whole chunk.
        IOB      Two labels for chunk type X, B-X for chunk begining and I-X for chunk inside.
        IOE      Two labels for chunk type X, E-X for chunk ending and I-X for chunk inside.
        IOBES    Four labels for chunk type X, B-X for chunk begining, I-X for chunk inside, E-X for chunk end and S-X for single word chunk.

    To make it clear, let's illustrate by an NER example.
    Assuming that there are three named entity types including ORG, PER and LOC which are called 'chunk type' here,
    if 'IOB' scheme were used, the label set will be extended to a set including B-ORG, I-ORG, B-PER, I-PER, B-LOC, I-LOC and O,
    in which B-ORG for begining of ORG and I-ORG for inside of ORG.
    Prefixes which are called 'tag type' here are added to chunk types and there are two tag types including B and I.
    Of course, the training data should be labeled accordingly.

    - Mapping is done correctly by the listed equations and assigning protocol.

    The following table are equations to extract tag type and chunk type from a label.

    .. code-block:: text

        tagType = label % numTagType
        chunkType = label / numTagType
        otherChunkType = numChunkTypes

    The following table shows the mapping rule between tagType and tag type in each scheme.

    .. code-block:: text

        Scheme Begin Inside End   Single
        plain  0     -      -     -
        IOB    0     1      -     -
        IOE    -     0      1     -
        IOBES  0     1      2     3

    Continue the NER example, and the label dict should look like this to satify above equations:

    .. code-block:: text

        B-ORG  0
        I-ORG  1
        B-PER  2
        I-PER  3
        B-LOC  4
        I-LOC  5
        O      6

    In this example, chunkType has three values: 0 for ORG, 1 for PER, 2 for LOC, because the scheme is
    "IOB" so tagType has two values: 0 for B and 1 for I.
    Here we will use I-LOC to explain the above mapping rules in detail.
    For I-LOC, the label id is 5, so we can get tagType=1 and chunkType=2, which means I-LOC is a part of NER chunk LOC
    and the tag is I.

    The simple usage is:

    .. code-block:: python

       eval = chunk_evaluator(input, label, chunk_scheme, num_chunk_types)


    :param input: The input layers.
    :type input: LayerOutput
    :param label: An input layer containing the ground truth label.
    :type label: LayerOutput
    :param chunk_scheme: The labelling schemes support 4 types. It is one of
                         "IOB", "IOE", "IOBES", "plain". It is required.
    :type chunk_scheme: basestring
    :param num_chunk_types: number of chunk types other than "other"
    :param name: The Evaluator name, it is optional.
    :type name: basename|None
    :param excluded_chunk_types: chunks of these types are not considered
    :type excluded_chunk_types: list of integer|None
    """
    evaluator_base(
        name=name,
        type="chunk",
        input=input,
        label=label,
        chunk_scheme=chunk_scheme,
        num_chunk_types=num_chunk_types,
        excluded_chunk_types=excluded_chunk_types, )


@evaluator(EvaluatorAttribute.FOR_UTILS)
@wrap_name_default()
def sum_evaluator(
        input,
        name=None,
        weight=None, ):
    """
    An Evaluator to sum the result of input.

    The simple usage:

    .. code-block:: python

       eval = sum_evaluator(input)

    :param name: Evaluator name.
    :type name: None|basestring
    :param input: Input Layer name.
    :type input: LayerOutput
    :param weight: Weight Layer name. It should be a matrix with size
                  [sample_num, 1]. (TODO, explaination)
    :type weight: LayerOutput
    """
    evaluator_base(name=name, type="sum", input=input, weight=weight)


@evaluator(EvaluatorAttribute.FOR_UTILS)
@wrap_name_default()
def column_sum_evaluator(
        input,
        name=None,
        weight=None, ):
    """
    This Evaluator is used to sum the last column of input.

    The simple usage is:

    .. code-block:: python

       eval = column_sum_evaluator(input, label)

    :param name: Evaluator name.
    :type name: None|basestring
    :param input: Input Layer name.
    :type input: LayerOutput
    """
    evaluator_base(
        name=name, type="last-column-sum", input=input, weight=weight)


"""
The following are printer Evaluators which are usually used to
print the result, like value or gradient of input layers, the
results generated in machine translation, the classification error etc.
"""


@evaluator(EvaluatorAttribute.FOR_PRINT)
@wrap_name_default()
def value_printer_evaluator(
        input,
        name=None, ):
    """
    This Evaluator is used to print the values of input layers. It contains
    one or more input layers.

    The simple usage is:

    .. code-block:: python

       eval = value_printer_evaluator(input)

    :param input: One or more input layers.
    :type input: LayerOutput|list
    :param name: Evaluator name.
    :type name: None|basestring
    """
    evaluator_base(name=name, type="value_printer", input=input)


@evaluator(EvaluatorAttribute.FOR_PRINT)
@wrap_name_default()
def gradient_printer_evaluator(
        input,
        name=None, ):
    """
    This Evaluator is used to print the gradient of input layers. It contains
    one or more input layers.

    The simple usage is:

    .. code-block:: python

       eval = gradient_printer_evaluator(input)

    :param input: One or more input layers.
    :type input: LayerOutput|list
    :param name: Evaluator name.
    :type name: None|basestring
    """
    evaluator_base(name=name, type="gradient_printer", input=input)


@evaluator(EvaluatorAttribute.FOR_PRINT)
@wrap_name_default()
def maxid_printer_evaluator(
        input,
        num_results=None,
        name=None, ):
    """
    This Evaluator is used to print maximum top k values and their indexes
    of each row of input layers. It contains one or more input layers.
    k is specified by num_results.

    The simple usage is:

    .. code-block:: python

       eval = maxid_printer_evaluator(input)

    :param input: Input Layer name.
    :type input: LayerOutput|list
    :param num_results: This number is used to specify the top k numbers.
                        It is 1 by default.
    :type num_results: int.
    :param name: Evaluator name.
    :type name: None|basestring
    """
    evaluator_base(
        name=name, type="max_id_printer", input=input, num_results=num_results)


@evaluator(EvaluatorAttribute.FOR_PRINT)
@wrap_name_default()
def maxframe_printer_evaluator(
        input,
        num_results=None,
        name=None, ):
    """
    This Evaluator is used to print the top k frames of each input layers.
    The input layers should contain sequences info or sequences type.
    k is specified by num_results.
    It contains one or more input layers.

    Note:
        The width of each frame is 1.

    The simple usage is:

    .. code-block:: python

       eval = maxframe_printer_evaluator(input)

    :param input: Input Layer name.
    :type input: LayerOutput|list
    :param name: Evaluator name.
    :type name: None|basestring
    """
    evaluator_base(
        name=name,
        type="max_frame_printer",
        input=input,
        num_results=num_results)


@evaluator(EvaluatorAttribute.FOR_PRINT)
@wrap_name_default()
def seqtext_printer_evaluator(
        input,
        result_file,
        id_input=None,
        dict_file=None,
        delimited=None,
        name=None, ):
    """
    Sequence text printer will print text according to index matrix and a
    dictionary. There can be multiple input to this layer:

    1. If there is no id_input, the input must be a matrix containing
    the sequence of indices;

    2. If there is id_input, it should be ids, and interpreted as sample ids.

    The output format will be:

    1. sequence without sub-sequence, and there is probability.

    .. code-block:: python

         id \t prob space_seperated_tokens_from_dictionary_according_to_seq

    2. sequence without sub-sequence, and there is not probability.

    .. code-block:: python

         id \t space_seperated_tokens_from_dictionary_according_to_seq

    3. sequence with sub-sequence, and there is not probability.

    .. code-block:: python

         id \t space_seperated_tokens_from_dictionary_according_to_sub_seq
         \t \t space_seperated_tokens_from_dictionary_according_to_sub_seq
         ...

    Typically SequenceTextPrinter layer takes output of maxid or RecurrentGroup
    with maxid (when generating) as an input.

    The simple usage is:

    .. code-block:: python

       eval = seqtext_printer_evaluator(input=maxid_layer,
                                        id_input=sample_id,
                                        dict_file=dict_file,
                                        result_file=result_file)

    :param input: Input Layer name.
    :type input: LayerOutput|list
    :param result_file: Path of the file to store the generated results.
    :type result_file: basestring
    :param id_input: Index of the input sequence, and the specified index will
                     be prited in the gereated results. This an optional
                     parameter.
    :type id_input: LayerOutput
    :param dict_file: Path of dictionary. This is an optional parameter.
                      Every line is a word in the dictionary with
                      (line number - 1) as the word index.
                      If this parameter is set to None, or to an empty string,
                      only word index are printed in the generated results.
    :type dict_file: basestring
    :param delimited: Whether to use space to separate output tokens.
                Default is True. No space is added if set to False.
    :type delimited: bool
    :param name: Evaluator name.
    :type name: None|basestring
    :return: The seq_text_printer that prints the generated sequence to a file.
    :rtype: evaluator
    """
    assert isinstance(result_file, basestring)
    if id_input is None:
        inputs = [input]
    else:
        inputs = [id_input, input]
        input.parents.append(id_input)

    evaluator_base(
        name=name,
        type="seq_text_printer",
        input=inputs,
        dict_file=dict_file,
        result_file=result_file,
        delimited=delimited)


@evaluator(EvaluatorAttribute.FOR_PRINT)
@wrap_name_default()
def classification_error_printer_evaluator(
        input,
        label,
        threshold=0.5,
        name=None, ):
    """
    This Evaluator is used to print the classification error of each sample.

    The simple usage is:

    .. code-block:: python

       eval = classification_error_printer_evaluator(input)

    :param input: Input layer.
    :type input: LayerOutput
    :param label: Input label layer.
    :type label: LayerOutput
    :param name: Evaluator name.
    :type name: None|basestring
    """
    evaluator_base(
        name=name,
        type="classification_error_printer",
        input=input,
        label=label,
        classification_threshold=threshold)
