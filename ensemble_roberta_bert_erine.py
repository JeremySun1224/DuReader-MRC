import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# ALL_MODELS = sum(
#     (tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES),
#     (),
# )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def load_and_cache_examples(
    args, model_path1, tokenizer, evaluate=False, output_examples=False
):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_path1.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and (
            (evaluate and not args.predict_file)
            or (not evaluate and not args.train_file)
        ):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError(
                    "If not data_dir is specified, tensorflow_datasets needs to be installed."
                )

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(
                tfds_examples, evaluate=evaluate
            )
        else:
            processor = (
                SquadV2Processor()
                if args.version_2_with_negative
                else SquadV1Processor()
            )
            if evaluate:
                examples = processor.get_dev_examples(
                    args.data_dir, filename=args.predict_file
                )
            else:
                examples = processor.get_train_examples(
                    args.data_dir, filename=args.train_file
                )

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(
                {"features": features, "dataset": dataset, "examples": examples},
                cached_features_file,
            )

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def evaluate(args, model_path1, model_path2, model_path3, 
    model1, model2, model3, 
    tokenizer1, tokenizer2, tokenizer3, prefix=""):
    # dataset, examples, features = load_and_cache_examples(
    #     args, model_path1, tokenizer, evaluate=True, output_examples=True
    # )
    dataset1, examples1, features1 = load_and_cache_examples(
        args, model_path1, tokenizer1, evaluate=True, output_examples=True
    )
    dataset2, examples2, features2 = load_and_cache_examples(
        args, model_path2, tokenizer2, evaluate=True, output_examples=True
    )
    dataset3, examples3, features3 = load_and_cache_examples(
        args, model_path3, tokenizer3, evaluate=True, output_examples=True
    )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly

    eval_sampler1 = SequentialSampler(dataset1)
    eval_dataloader1 = DataLoader(
        dataset1, sampler=eval_sampler1, batch_size=args.eval_batch_size
    )
    eval_sampler2 = SequentialSampler(dataset2)
    eval_dataloader2 = DataLoader(
        dataset2, sampler=eval_sampler2, batch_size=args.eval_batch_size
    )
    eval_sampler3 = SequentialSampler(dataset3)
    eval_dataloader3 = DataLoader(
        dataset3, sampler=eval_sampler3, batch_size=args.eval_batch_size
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model1, torch.nn.DataParallel):
        model1 = torch.nn.DataParallel(model1)

    if args.n_gpu > 1 and not isinstance(model2, torch.nn.DataParallel):
        model2 = torch.nn.DataParallel(model2)

    if args.n_gpu > 1 and not isinstance(model3, torch.nn.DataParallel):
        model3 = torch.nn.DataParallel(model3)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset1))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()
    model_result = {}
    for batch in tqdm(eval_dataloader1, desc="Evaluating"):
        model1.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            feature_indices = batch[3]
            outputs1 = model1(**inputs)
        for i, feature_index in enumerate(feature_indices):
            
            eval_feature = features1[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            # logger.info("unique_id1 = %s", str(unique_id))
            output1 = [to_list(output1[i]) for output1 in outputs1]
            start_logits1, end_logits1 = output1
            model_result[unique_id] = [(start_logits1, end_logits1)]

    for batch in tqdm(eval_dataloader2, desc="Evaluating"):
        model2.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            feature_indices = batch[3]
            outputs2 = model2(**inputs)
        for i, feature_index in enumerate(feature_indices):
            eval_feature = features2[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            # logger.info("unique_id2 = %s", str(unique_id))
            output2 = [to_list(output2[i]) for output2 in outputs2]
            start_logits2, end_logits2 = output2
            model_result[unique_id].append((start_logits2, end_logits2))

    for batch in tqdm(eval_dataloader3, desc="Evaluating"):
        model3.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            feature_indices = batch[3]
            outputs3 = model3(**inputs)
        for i, feature_index in enumerate(feature_indices):
            eval_feature = features3[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            # logger.info("unique_id3 = %s", str(unique_id))
            output3 = [to_list(output3[i]) for output3 in outputs3]
            start_logits3, end_logits3 = output3
            model_result[unique_id].append((start_logits3, end_logits3))

    weights = [0.4, 0.2, 0.4]
    start_logits = []
    end_logits = []
    # logger.info("model_result = %s", str(model_result))
    for k, v in model_result.items():
        # logger.info("model_result len = %s", str(len(v)))
        # logger.info("model_result_v0 len = %s", str(len(v[0])))
        # logger.info("model_result_v1 len = %s", str(len(v[1])))
        # logger.info("model_result_v2 len = %s", str(len(v[2])))
        if len(v) == 2:
            start_logits = [weights[0] * log1 + weights[1] * log2
                for log1, log2 in zip(v[0][0], v[1][0])]
            end_logits = [weights[0] * log1 + weights[1] * log2
                for log1, log2 in zip(v[0][1], v[1][1])]
        else:
            start_logits = [weights[0] * log1 + weights[1] * log2 + weights[2] * log3 
                for log1, log2, log3 in zip(v[0][0], v[1][0], v[2][0])]
            end_logits = [weights[0] * log1 + weights[1] * log2 + weights[2] * log3
                for log1, log2, log3 in zip(v[0][1], v[1][1], v[2][1])]
        result = SquadResult(k, start_logits, end_logits)
        all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info(
        "  Evaluation done in total %f secs (%f sec per example)",
        evalTime,
        evalTime / len(dataset1),
    )

    # Compute predictions
    output_prediction_file = os.path.join(
        args.output_dir, "predictions_{}.json".format(prefix)
    )
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions_{}.json".format(prefix)
    )

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds_{}.json".format(prefix)
        )
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples1,
        features1,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer1,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples1, predictions)
    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    # parser.add_argument(
    #     "--model_name_or_path",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="Path to pre-trained model or shortcut name selected in the list: "
    #     + ", ".join(ALL_MODELS),
    # )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="Can be used for distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="Can be used for distant debugging."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="multiple threads for converting example to features",
    )
    # parser.add_argument(
    #     "--gru_layers", type=int, default=1, help="number of layers in GRU"
    # )
    # parser.add_argument(
    #     "--gru_hidden_size", type=int, default=256, help="hidden size in GRU"
    # )
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    # args.no_cuda = True
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    model_path1 = "/content/dureader/output/chinese_bert_wwm_ext_pytorch_ro/DuReader/checkpoint-2000/"
    model_path2 = "/content/dureader/output/chinese_wwm_ext_pytorch/DuReader/checkpoint-2000/"
    model_path3 = "/content/dureader/output/chinese_ernie_base/DuReader/checkpoint-2000/"

    # config1 = AutoConfig.from_pretrained(
    #     model_path1,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )
    tokenizer1 = AutoTokenizer.from_pretrained(
        model_path1,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer2 = AutoTokenizer.from_pretrained(
        model_path2,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer3 = AutoTokenizer.from_pretrained(
        model_path3,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # model1 = AutoModelForQuestionAnswering.from_pretrained(
    #     model_path1,
    #     from_tf=bool(".ckpt" in args.model_name_or_path),
    #     config=config,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    # model1.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    logger.info("Loading checkpoint %s for evaluation", model_path1)
    global_step = ""
    # Reload the model
    model1 = AutoModelForQuestionAnswering.from_pretrained(
        model_path1
    )  # , force_download=True)
    model1.to(args.device)

    model2 = AutoModelForQuestionAnswering.from_pretrained(
        model_path2
    )  # , force_download=True)
    model2.to(args.device)

    model3 = AutoModelForQuestionAnswering.from_pretrained(
        model_path3
    )  # , force_download=True)
    model3.to(args.device)
    # Evaluate
    result = evaluate(args, model_path1, model_path2, model_path3,
        model1, model2, model3, tokenizer1, tokenizer2, tokenizer3, prefix="")

    result = dict(
        (k + ("_{}".format(global_step) if global_step else ""), v)
        for k, v in result.items()
    )
    results.update(result)

    # logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
