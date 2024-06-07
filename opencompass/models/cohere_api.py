import asyncio  # for running API calls concurrently
import atexit
import copy
import json  # for saving results to a jsonl file
import os
from pathlib import Path
import re  # for matching endpoint from request URL
import signal
import tempfile
import jieba
import time  # for sleeping after rate limit is hit
from collections import defaultdict
from requests import get as get_request
from opencompass.utils.logging import get_logger

# Openai parallel processor imports
from dataclasses import (
    dataclass,
    field,
)

# for storing API inputs, outputs, and metadata
from importlib.util import find_spec
from typing import Dict, List, Literal, Optional, Tuple, Union

import aiohttp  # for making API calls concurrently
import tiktoken  # for counting tokens
from tokenizers import Tokenizer
from tqdm import tqdm


from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]

COHERE_API_BASE="https://api.cohere.com/v1/chat"

async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    tokenizer: Tokenizer,
    max_attempts: int,
    logger
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # infer API endpoint and construct request header
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logger.debug("Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logger.debug("File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logger.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, tokenizer
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logger.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logger.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logger.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logger.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logger.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logger.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        
        logger = get_logger("INFO")
        """Calls the OpenAI API and saves results."""
        logger.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                status = response.status
                response = await response.json()
            if status < 200 or status >= 300:
                logger.warning(
                    f"Request {self.task_id} failed with error code {str(status)}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if status == 429:
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logger.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logger.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logger.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions

def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    tokenizer: Tokenizer,
):
    completion_tokens = request_json.get("max_tokens", 256)

    prompt_tokens = 0
    if history := request_json.get("chat_history"):
        for message in history:
            prompt_tokens += len(tokenizer.encode(message["message"], add_special_tokens=False).tokens)
    
    prompt_tokens += len(tokenizer.encode(request_json["message"]).tokens)

    return prompt_tokens + completion_tokens


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def process_chat_request(messages, model, idx, **kwargs):
    request = {
        "message": messages,
        "model": model,
        "metadata": {"idx": idx},
    }

    request.update(kwargs)

    return json.dumps(request)


def get_result(response, ctxlen: int) -> Tuple[float, bool]:
    """Process results from OpenAI API response.

    :param response: dict
        OpenAI API Response
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: np.array
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    is_greedy = True
    logprobs = response.logprobs.token_logprobs
    continuation_logprobs = sum(logprobs[ctxlen:])

    for i in range(ctxlen, len(response.logprobs.token_logprobs)):
        token = response.logprobs.token_logprobs[i]
        top_tokens = response.logprobs.top_logprobs[i]
        top_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
        if top_token != token:
            is_greedy = False
            break

    return continuation_logprobs, is_greedy



@MODELS.register_module()
class Cohere(BaseAPIModel):
    """Model wrapper around Cohere's models.

    Args:
        path (str): The name of Cohere's model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        key (str or List[str]): OpenAI key(s). In particular, when it
            is set to "ENV", the key will be fetched from the environment
            variable $OPENAI_API_KEY, as how openai defaults to be. If it's a
            list, the keys will be used in round-robin manner. Defaults to
            'ENV'.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        cohere_api_base (str): The base url of OpenAI's API. Defaults to
            'https://api.cohere.com/v1/chat'.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'front','mid' and 'rear' represents the part
            of input to truncate. Defaults to 'none'.
        temperature (float, optional): What sampling temperature to use.
            If not None, will override the temperature in the `generate()`
            call. Defaults to None.
        tokenizer_path (str, optional): The path to the tokenizer. Use path if
            'tokenizer_path' is None, otherwise use the 'tokenizer_path'.
            Defaults to None.
    """

    is_api: bool = True

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 meta_template: Optional[Dict] = None,
                 mode: str = 'none',
                 temperature: Optional[float] = None,):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry)

        self.temperature = temperature
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode

        if isinstance(key, str):
            if key == 'ENV':
                if 'COHERE_API_KEY' not in os.environ:
                    raise ValueError('COHERE API key is not set.')
                self.keys = os.getenv('COHERE_API_KEY').split(',')
            else:
                self.keys = [key]
        else:
            self.keys = key

        self.key_ctr = 0
        self.path = path

        tokenizer_url = ""

        retry_tokenizer = 5
        
        while True:
            model_url = f"https://api.cohere.com/v1/models/{self.path}"
            if retry_tokenizer == 0:
                raise ConnectionError(f"Unable to get tokenizer from {model_url}.")
            model_info = get_request(
                url=model_url,
                headers={
                    "accept": "application/json",
                    "Authorization": f"BEARER {self.keys[self.key_ctr]}"
                }
            )
            if model_info.status_code == 200:
                tokenizer_url = model_info.json()["tokenizer_url"]
                break
            else:
                retry_tokenizer -= 1

                time.sleep(5)


        tokenizer_config = get_request(tokenizer_url)
        self.tokenizer: Tokenizer = Tokenizer.from_str(tokenizer_config.text)

    def generate(self,
                inputs: List[PromptType],
                max_out_len: int = 512,
                temperature: float = 0.7):
        
        if self.temperature is not None:
                temperature = self.temperature

        temp_dir = tempfile.gettempdir()
        requests_file_path = os.path.join(
            temp_dir, "opencompass_requests.jsonl",
        )
        responses_file_path = os.path.join(
            temp_dir, "opencompass_responses.jsonl",
        )

        def clean_up_requests(signum=None, frame=None):
                if os.path.exists(requests_file_path):
                    os.remove(requests_file_path)
                    print("Cached requests temp file deleted.")
                if os.path.exists(responses_file_path):
                    os.remove(responses_file_path)
                    print("Cached responses temp file deleted.")
                if signum is not None:
                    exit(0)  # Only exit if called by a signal
        
        atexit.register(clean_up_requests)
        signal.signal(signal.SIGINT, clean_up_requests)
        signal.signal(signal.SIGTERM, clean_up_requests)

        with open(requests_file_path, "w") as f:
                for idx, input in enumerate(inputs):
                    data = self.get_request_data(input=input, max_out_len=max_out_len, temperature=temperature)

                    data = data | {"metadata": {"idx" : idx} }

                    f.write(json.dumps(data) + "\n")
        
        max_requests_per_minute = 10_000
        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=str(requests_file_path),
                save_filepath=str(responses_file_path),
                request_url=str("https://api.cohere.com/v1/chat"),
                api_key=str(self.keys[self.key_ctr]),
                max_requests_per_minute=float(
                    max_requests_per_minute * 0.75
                ),  # *0.75 to leave some headroom
                max_tokens_per_minute=float(2000000),
                tokenizer=self.tokenizer,
                max_attempts=10,
                logger=get_logger('INFO')
            )
        )

        with open(responses_file_path, "r") as f:
            lines = f.readlines()
        
        results = []

        input_toks = 0
        output_toks = 0

        for line in lines:
            response_object = json.loads(line)
            context = response_object[0]["message"]
            response = response_object[1]["text"]
            idx = response_object[2]["idx"]

            input_toks += int(response_object[1]["meta"]["tokens"]["input_tokens"])
            output_toks += int(response_object[1]["meta"]["tokens"]["output_tokens"])

            results.append((idx, response))
            
        results.sort(key=lambda x: x[0])

        clean_up_requests()

        cwd = os.getcwd()

        with open("TOKEN_COUNTER.json", "r+") as f:
            token_counter = json.load(f)
            total_input_toks = int(token_counter["input_toks"]) + input_toks
            total_output_toks = int(token_counter["output_toks"]) + output_toks
            print(total_input_toks, total_output_toks)
            f.seek(0)
            f.truncate()
            f.write(json.dumps({"input_toks": total_input_toks, "output_toks": total_output_toks}))

        return [s for _, s in results]




    def get_request_data(self, input: PromptType, max_out_len: int,
                    temperature: float):
                # max num token for gpt-3.5-turbo is 4097
        context_window = 128_000

        # will leave 100 tokens as prompt buffer, triggered if input is str
        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)


        data = {}
        if isinstance(input, str):
            data["message"] = input
        else:
            data["message"] = input[-1]['prompt']
            history = []
            for item in input[:-1]:
                msg = {"message": item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'USER'
                elif item['role'] == 'BOT':
                    msg['role'] = 'CHATBOT'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'SYSTEM'
                history.append(msg)
            data["chat_history"] = history

        try:
            max_out_len = min(
                max_out_len,
                context_window - self.get_token_len(str(input)) - 100)
        except KeyError:
            max_out_len = max_out_len
        if max_out_len <= 0:
            return ''

        data = data | dict(
                    model=self.path,
                    max_tokens=max_out_len,
                    temperature=temperature,
        )
        
        return data

    def bin_trim(self, prompt: str, num_token: int) -> str:
        """Get a suffix of prompt which is no longer than num_token tokens.

        Args:
            prompt (str): Input string.
            num_token (int): The upper bound of token numbers.

        Returns:
            str: The trimmed prompt.
        """
        token_len = self.get_token_len(prompt)
        if token_len <= num_token:
            return prompt
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if pattern.search(prompt):
            words = list(jieba.cut(prompt, cut_all=False))
            sep = ''
        else:
            words = prompt.split(' ')
            sep = ' '

        l, r = 1, len(words)
        while l + 2 < r:
            mid = (l + r) // 2
            if self.mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif self.mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
            elif self.mode == 'rear':
                cur_prompt = sep.join(words[:mid])

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid  # noqa: E741
            else:
                r = mid

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt
    
    def get_token_len(self, prompt: str) -> int:

        return len(self.tokenizer.encode(prompt).tokens)
        