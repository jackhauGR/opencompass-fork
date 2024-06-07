# flake8: noqa: E501
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import os
from dotenv import load_dotenv, find_dotenv

import requests
import asyncio
import atexit
from dataclasses import dataclass, field

from opencompass.utils.logging import get_logger
import tempfile
import signal
import aiohttp
import tiktoken
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str, float]


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
    logger
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # infer API endpoint and construct request header
    api_endpoint = request_url
    request_header = {'content-type': 'application/json',}

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
                                    request_json, api_endpoint, token_encoding_name
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

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

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
        logger = get_logger()
        """Calls the OpenAI API and saves results."""
        logger.info(f"Starting request #{self.task_id}")
        error = None

        try:

            async with session.post(
                url=request_url, headers=request_header, data=json.dumps(self.request_json)
            ) as response:
                response = await response.json()
            if "error" in response:
                logger.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "") or response["error"].get("code") ==  403:
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


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)

    max_tokens = request_json.get("generationConfig", {}).get("maxOutputTokens", 15)
    n = request_json.get("generationConfig", {}).get("candidate_count", 1)

    completion_tokens = n * max_tokens

    prompt = request_json["contents"][0]["parts"][0]["text"]

    prompt_tokens = len(encoding.encode(prompt))

    return prompt_tokens + completion_tokens


class Gemini(BaseAPIModel):
    """Model wrapper around Gemini models.

    Documentation:

    Args:
        path (str): The name of Gemini model.
            e.g. `gemini-pro`
        key (str): Authorization key.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        max_seq_len (int): Unused here.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """

    def __init__(
        self,
        key: str,
        path: str,
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: float = 10.0,
    ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        
        if isinstance(key, str):
            if key == "ENV":
                
                if 'GOOGLE_API_KEY' not in os.environ:
                    raise ValueError('Google API key is not set.')
                self.key = os.getenv('GOOGLE_API_KEY')
            else:
                self.key = [key]
        else:
            self.key = key

        self.url = f'https://generativelanguage.googleapis.com/v1/models/{path}:generateContent?key={self.key}'
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.headers = {
            'content-type': 'application/json',
        }

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        # with ThreadPoolExecutor() as executor:
        #     results = list(
        #         executor.map(self._generate, inputs,
        #                      [max_out_len] * len(inputs)))
        # self.flush()
        # return results

        print("hello00000000000000000000000")

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
                data = self.get_request_data(input=input, max_out_len=max_out_len, temperature=self.temperature)

                data = data | {"metadata": {"idx" : idx} }

                f.write(json.dumps(data) + "\n")

        max_requests_per_minute = 300
        max_tokens_per_minute = 100000

        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=str(requests_file_path),
                save_filepath=str(responses_file_path),
                request_url=str(self.url),
                api_key=str(self.key),
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=self.retry,
                logging_level=int(30),
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

            response = response_object[1]["candidates"][0].get("content", {"parts": [{"text": ""}]})["parts"][0]["text"]
            idx = response_object[2]["idx"]

            input_toks += int(response_object[1]["usageMetadata"].get("promptTokenCount", 0))
            output_toks += int(response_object[1]["usageMetadata"].get("candidatesTokenCount", 0))

            results.append((idx, response))
            
        results.sort(key=lambda x: x[0])

        clean_up_requests()

        with open("TOKEN_COUNTER.json", "r+") as f:
            token_counter = json.load(f)
            total_input_toks = int(token_counter["input_toks"]) + input_toks
            total_output_toks = int(token_counter["output_toks"]) + output_toks
            print(total_input_toks, total_output_toks)
            f.seek(0)
            f.truncate()
            f.write(json.dumps({"input_toks": total_input_toks, "output_toks": total_output_toks}))

        return [s for _, s in results]
    
    def get_request_data(self, input: PromptType, max_out_len: int, temperature: float):
        
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{'role': 'user', 'parts': [{'text': input}]}]
        else:
            messages = []
            system_prompt = None
            for item in input:
                if item['role'] == 'SYSTEM':
                    system_prompt = item['prompt']
            for item in input:
                if system_prompt is not None:
                    msg = {
                        'parts': [{
                            'text': system_prompt + '\n' + item['prompt']
                        }]
                    }
                else:
                    msg = {'parts': [{'text': item['prompt']}]}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                    messages.append(msg)
                elif item['role'] == 'BOT':
                    msg['role'] = 'model'
                    messages.append(msg)
                elif item['role'] == 'SYSTEM':
                    pass

            # model can be response with user and system
            # when it comes with agent involved.
            assert msg['role'] in ['user', 'system']

        data = {
            'model':
            self.path,
            'contents':
            messages,
            'safetySettings': [
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE'
                },
            ],
            'generationConfig': {
                'candidate_count': 1,
                'temperature': temperature,
                'maxOutputTokens': max_out_len,
                'topP': self.top_p,
                'topK': self.top_k
            }
        }

        return data


    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{'role': 'user', 'parts': [{'text': input}]}]
        else:
            messages = []
            system_prompt = None
            for item in input:
                if item['role'] == 'SYSTEM':
                    system_prompt = item['prompt']
            for item in input:
                if system_prompt is not None:
                    msg = {
                        'parts': [{
                            'text': system_prompt + '\n' + item['prompt']
                        }]
                    }
                else:
                    msg = {'parts': [{'text': item['prompt']}]}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                    messages.append(msg)
                elif item['role'] == 'BOT':
                    msg['role'] = 'model'
                    messages.append(msg)
                elif item['role'] == 'SYSTEM':
                    pass

            # model can be response with user and system
            # when it comes with agent involved.
            assert msg['role'] in ['user', 'system']

        data = {
            'model':
            self.path,
            'contents':
            messages,
            'safetySettings': [
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE'
                },
            ],
            'generationConfig': {
                'candidate_count': 1,
                'temperature': self.temperature,
                'maxOutputTokens': 2048,
                'topP': self.top_p,
                'topK': self.top_k
            }
        }

        for _ in range(self.retry):
            self.wait()
            raw_response = requests.post(self.url,
                                         headers=self.headers,
                                         data=json.dumps(data))
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
                time.sleep(1)
                continue
            if raw_response.status_code == 200:

                if 'candidates' not in response:
                    self.logger.error(response)
                else:
                    if 'content' not in response['candidates'][0]:
                        return "Due to Google's restrictive policies, I am unable to respond to this question."
                    else:
                        return response['candidates'][0]['content']['parts'][0][
                            'text'].strip()
            self.logger.error(response["error"]["message"])
            self.logger.error(response)
            time.sleep(1)

        raise RuntimeError('API call failed.')
