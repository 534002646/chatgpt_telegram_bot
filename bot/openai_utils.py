import config

import tiktoken
from openai import AsyncOpenAI
import io
import base64
import logging
from PIL import Image

logger = logging.getLogger(__name__)

openai = AsyncOpenAI(
    api_key=config.openai_api_key,
)

class ChatGPT:
    def __init__(self, model=config.default_model):
        assert model in config.models["available_text_models"], f"未知模型： {model}"
        self.model = model

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}]
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model=config.default_model):
        try:
            encoding = tiktoken.encoding_for_model(model)
            if "gpt-3.5" in model:
                tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                tokens_per_name = -1  # if there's a name, the role is omitted
            elif "gpt-4" in model:
                tokens_per_message = 3
                tokens_per_name = 1
            else:
                raise ValueError(f"未知模型： {model}")
            # input
            n_input_tokens = 0
            for message in messages:
                n_input_tokens += tokens_per_message
                for key, value in message.items():
                    if key == 'content':
                        if isinstance(value, str):
                            n_input_tokens += len(encoding.encode(value))
                        else:
                            for msg in value:
                                if msg['type'] == 'image_url':
                                    image = decode_image(msg['image_url']['url'])
                                    n_input_tokens += self.__count_tokens_vision(image, model)
                                else:
                                    n_input_tokens += len(encoding.encode(msg['text']))
                    else:
                        n_input_tokens += len(encoding.encode(value))
                        if key == "name":
                            n_input_tokens += tokens_per_name

            n_input_tokens += 3
            # output
            n_output_tokens = 1 + len(encoding.encode(str(answer)))
            return n_input_tokens, n_output_tokens
        except Exception as e:  # too many tokens
           logger.error(f"统计tokens出现异常. {e}")
        return 0,0
    
    def __count_tokens_vision(self, image_bytes: bytes, model=config.default_model) -> int:
        image_file = io.BytesIO(image_bytes)
        image = Image.open(image_file)
        if "gpt-4" not in model:
            return 0
        w, h = image.size
        if w > h: w, h = h, w
        base_tokens = 85
        detail = config.vision_detail
        if detail == 'low':
            return base_tokens
        elif detail == 'high' or detail == 'auto': # assuming worst cost for auto
            f = max(w / 768, h / 2048)
            if f > 1:
                w, h = int(w / f), int(h / f)
            tw, th = (w + 511) // 512, (h + 511) // 512
            tiles = tw * th
            num_tokens = base_tokens + tiles * 170
            return num_tokens
        else:
           return 0
        
    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in config.models["available_text_models"]:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    common_args = {
                        'model': self.model,
                        'messages': messages,
                        'temperature': 1,
                        'n': 1,
                        'max_tokens': 2400,
                        'presence_penalty': 0,
                        'frequency_penalty': 0,
                        'stream': True
                    }
                    # logger.error(f"提问内容：{common_args}")
                    r = await openai.chat.completions.create(**common_args)
                    answer = ""
                    async for r_item in r:
                        delta = r_item.choices[0].delta
                        if delta.content:
                            answer += delta.content
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

                answer = self._postprocess_answer(answer)
                # logger.error(f"回答内容：{answer}")
            except Exception as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise e
                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

#语音翻译
async def transcribe_audio(audio_buf) -> str:
    r = await openai.audio.transcriptions.create(model="whisper-1", file=audio_buf)
    return r.text or ""

#生成图片
async def generate_images(prompt: str, model: str):
    r = await openai.images.generate(
        prompt=prompt, 
        n=config.image_count,
        model=model,
        quality=config.image_quality,
        style=config.image_style,
        size=config.image_size)
    return [item.url for item in r.data]

#生成语言
async def generate_audio(text: str, model: str, current_audio_style: str):
    r = await openai.audio.speech.create(
        model=model,
        voice=current_audio_style,
        input=text,
        response_format=config.tts_voice_formats
    )
    temp_file = io.BytesIO()
    temp_file.write(r.read())
    temp_file.seek(0)
    return temp_file

def encode_image(fileobj):
    image = base64.b64encode(fileobj.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{image}'

def decode_image(imgbase64):
    image = imgbase64[len('data:image/jpeg;base64,'):]
    return base64.b64decode(image)