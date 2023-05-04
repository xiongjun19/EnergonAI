import argparse
import logging
import random
import asyncio
from typing import Optional

from energonai import QueueFullError, launch_engine
from energonai.model import opt_6B, opt_30B, opt_125M, opt_175B
from pydantic import BaseModel, Field
from transformers import GPT2Tokenizer

from batch import BatchManagerForGeneration
from cache import ListCache, MissCacheError


class GenerationTaskReq(BaseModel):
    max_tokens: int = Field(gt=0, le=256, example=64)
    prompt: str = Field(
        min_length=1, example='Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:')
    top_k: Optional[int] = Field(default=None, gt=0, example=50)
    top_p: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.5)
    temperature: Optional[float] = Field(default=None, gt=0.0, lt=1.0, example=0.7)


def get_model_fn(model_name: str):
    model_map = {
        'opt-125m': opt_125M,
        'opt-6.7b': opt_6B,
        'opt-30b': opt_30B,
        'opt-175b': opt_175B
    }
    return model_map[model_name]


def print_args(args: argparse.Namespace):
    print('\n==> Args:')
    for k, v in args.__dict__.items():
        print(f'{k} = {v}')


FIXED_CACHE_KEYS = [
    ('Question: What is the name of the largest continent on earth?\nAnswer: Asia\n\nQuestion: What is at the center of the solar system?\nAnswer:', 64),
    ('A chat between a salesman and a student.\n\nSalesman: Hi boy, are you looking for a new phone?\nStudent: Yes, my phone is not functioning well.\nSalesman: What is your budget? \nStudent: I have received my scholarship so I am fine with any phone.\nSalesman: Great, then perhaps this latest flagship phone is just right for you.', 64),
    ("English: I am happy today.\nChinese: 我今天很开心。\n\nEnglish: I am going to play basketball.\nChinese: 我一会去打篮球。\n\nEnglish: Let's celebrate our anniversary.\nChinese:", 64)
]

def main(tokenizer, engine, args):
    try:
        input_text = 'Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\n\nQuestion: What is the longest river on the earth?\nAnswer:'
        print("input is: ")
        print(input_text)
        inputs = tokenizer(input_text, padding="max_length", max_length=512)
        print(inputs)
        inputs['max_tokens'] = args.max_tokens
        inputs['top_k'] = 50
        inputs['top_p'] = 0.5
        inputs['temperature'] = 0.7
        uid = id(input_text)
        engine.submit(uid, inputs)
        output = asyncio.run(engine.wait(uid))
        output = tokenizer.decode(output, skip_special_tokens=True)
        print("output is: ")
        print(output)
        uid = id(input_text)
        engine.submit(uid, inputs)
        output = asyncio.run(engine.wait(uid))
        output = tokenizer.decode(output, skip_special_tokens=True)
        print("output2  is: ")
        print(output)


        return output
    finally:
        engine.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['opt-125m', 'opt-6.7b', 'opt-30b', 'opt-175b'])
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--master_host', default='localhost')
    parser.add_argument('--master_port', type=int, default=19990)
    parser.add_argument('--rpc_port', type=int, default=19980)
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--pipe_size', type=int, default=1)
    parser.add_argument('--queue_size', type=int, default=0)
    parser.add_argument('--http_host', default='0.0.0.0')
    parser.add_argument('--http_port', type=int, default=7070)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--cache_size', type=int, default=0)
    parser.add_argument('--cache_list_size', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=1)
    args = parser.parse_args()
    print_args(args)
    model_kwargs = {}
    if args.checkpoint is not None:
        model_kwargs['checkpoint'] = args.checkpoint

    logger = logging.getLogger(__name__)
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-30b')
    if args.cache_size > 0:
        cache = ListCache(args.cache_size, args.cache_list_size, fixed_keys=FIXED_CACHE_KEYS)
    else:
        cache = None
    engine = launch_engine(args.tp, 1, args.master_host, args.master_port, args.rpc_port, get_model_fn(args.model),
                           batch_manager=BatchManagerForGeneration(max_batch_size=args.max_batch_size,
                                                                   pad_token_id=tokenizer.pad_token_id),
                           pipe_size=args.pipe_size,
                           queue_size=args.queue_size,
                           **model_kwargs)
    outputs = main(tokenizer, engine, args)
    print("done")

