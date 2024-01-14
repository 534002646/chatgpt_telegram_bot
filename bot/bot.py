import io
import logging
import asyncio
import traceback
import html
import json
from datetime import datetime
from PIL import Image

import telegram
from telegram import (
    Update,
    User,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils


# setup
db = database.Database()
logger = logging.getLogger(__name__)

user_semaphores = {}
user_tasks = {}

HELP_MESSAGE = """Commands:
✅ /retry – 重新生成最后一个答案
✅ /new – 开始新对话
✅ /mode – 选择角色预设
✅ /img - 生成图片
✅ /audio - 生成语言
✅ /model – 显示模型选择
✅ /balance – 显示使用统计
✅ /cancel - 取消任务
✅ /help – 显示帮助
"""

HELP_GROUP_CHAT_MESSAGE = """您可以将机器人添加到任何<b>群聊</b>中，以帮助和娱乐参与者！

说明（请参阅下面的<b>视频</b>）：
1.将机器人添加到群聊
2.使其成为<b>管理员</b>，以便它可以看到消息（所有其他权限都可以限制）
3.你太棒了！

要在聊天中获得机器人的回复 – @<b>标记</b>它或<b>回复</b>其消息。
例如：“{bot_username} 写一首关于 Telegram 的诗”
"""


def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)

    if db.get_user_attribute(user.id, "current_model") is None:
        db.set_user_attribute(user.id, "current_model", config.default_model)
    if db.get_user_attribute(user.id, "current_audio_model") is None:
        db.set_user_attribute(user.id, "current_audio_model", config.default_audio_model)
    if db.get_user_attribute(user.id, "current_image_model") is None:
        db.set_user_attribute(user.id, "current_image_model", config.default_image_model)

    # back compatibility for n_used_tokens field
    n_used_tokens = db.get_user_attribute(user.id, "n_used_tokens")
    if isinstance(n_used_tokens, int) or isinstance(n_used_tokens, float):  # old format
        new_n_used_tokens = {
            config.default_model: {
                "n_input_tokens": 0,
                "n_output_tokens": n_used_tokens
            }
        }
        db.set_user_attribute(user.id, "n_used_tokens", new_n_used_tokens)

    # voice message transcription
    if db.get_user_attribute(user.id, "n_transcribed_seconds") is None:
        db.set_user_attribute(user.id, "n_transcribed_seconds", 0.0)
    # image generation
    for model_key in config.models["available_image_models"]:
        if db.get_user_attribute(user.id, model_key) is None:
            db.set_user_attribute(user.id, model_key, 0)
    # tts
    for model_key in config.models["available_audio_models"]:
        if db.get_user_attribute(user.id, model_key) is None:
            db.set_user_attribute(user.id, model_key, 0)


async def is_bot_mentioned(update: Update, context: CallbackContext):
     try:
         message = update.message

         if message.chat.type == "private":
             return True

         if message.text is not None and ("@" + context.bot.username) in message.text:
             return True

         if message.reply_to_message is not None:
             if message.reply_to_message.from_user.id == context.bot.id:
                 return True
     except:
         return True
     else:
         return False

# 开始
async def start_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id

    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)

    reply_text = "你好！ 我是使用 OpenAI API 实现的 <b>ChatGPT</b> 机器人 🤖\n\n"
    reply_text += HELP_MESSAGE
    await update.message.chat.send_action(action=ChatAction.TYPING)
    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)
    await show_chat_modes_handle(update, context)

# 帮助
async def help_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await update.message.chat.send_action(action=ChatAction.TYPING)
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)

# 群组帮助
async def help_group_chat_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update, context, update.message.from_user)
     user_id = update.message.from_user.id
     db.set_user_attribute(user_id, "last_interaction", datetime.now())

     text = HELP_GROUP_CHAT_MESSAGE.format(bot_username="@" + context.bot.username)
     await update.message.chat.send_action(action=ChatAction.TYPING)
     await update.message.reply_text(text, parse_mode=ParseMode.HTML)
     await update.message.reply_video(config.help_group_chat_video_path)

# 重新回答最后一条消息的
async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("没有消息可以重试 🤷‍♂️")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)


# 发送消息处理
async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return

    _message = message or update.message.text

    # remove bot mention (in group chats)
    if update.message.chat.type != "private":
        _message = _message.replace("@" + context.bot.username, "").strip()

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")

    async def message_handle_fn():
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"由于超时而启动新对话框 (<b>{config.chat_modes[chat_mode]['name']}</b> 模型) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # in case of CancelledError
        n_input_tokens, n_output_tokens = 0, 0
        current_model = db.get_user_attribute(user_id, "current_model")

        try:
            # send placeholder message to user
            placeholder_message = await update.message.reply_text("Loading...")

            if _message is None or len(_message) == 0:
                 await update.message.reply_text("🥲 您发送了<b>空消息</b>。 请再试一次！", parse_mode=ParseMode.HTML)
                 return

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[config.chat_modes[chat_mode]["parse_mode"]]

            chatgpt_instance = openai_utils.ChatGPT(model=current_model)
            gen = chatgpt_instance.send_message_stream(_message, dialog_messages=dialog_messages, chat_mode=chat_mode)
            prev_answer = ""
            await update.message.chat.send_action(action=ChatAction.TYPING)
            async for gen_item in gen:
                status, answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed = gen_item
                answer = answer[:4096]  # telegram message limit
                # update only when 100 new symbols are ready
                if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                    continue
                try:
                    await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id, parse_mode=parse_mode)
                except telegram.error.BadRequest as e:
                    if str(e).startswith("消息未修改"):
                        continue
                    else:
                        await context.bot.edit_message_text(answer, chat_id=placeholder_message.chat_id, message_id=placeholder_message.message_id)

                await asyncio.sleep(0.01)  # wait a bit to avoid flooding

                prev_answer = answer

            # update user data
            new_dialog_message = {"user": _message, "bot": answer, "date": datetime.now()}
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)

        except asyncio.CancelledError:
            # note: intermediate token updates only work when enable_message_streaming=True (config.yml)
            db.update_n_used_tokens(user_id, current_model, n_input_tokens, n_output_tokens)
            raise

        except Exception as e:
            error_text = f"完成过程中出了点问题， 原因：{e}"
            logger.error(error_text)
            await update.message.reply_text(error_text)
            return

        # send message if some messages were removed from the context
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>注意：</i>您当前的对话框太长，因此您的<b>第一条消息</b>已从上下文中删除。\n 发送 /new 命令以启动新对话框"
            else:
                text = f"✍️ <i>注意：</i> 您当前的对话框太长，因此<b>{n_first_dialog_messages_removed} 第一条消息</b>已从上下文中删除。\n 发送 /new 命令以启动新对话框"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async with user_semaphores[user_id]:
        task = asyncio.create_task(message_handle_fn())
        user_tasks[user_id] = task

        try:
            await task
        except asyncio.CancelledError:
            await update.message.reply_text("✅ 取消", parse_mode=ParseMode.HTML)
        else:
            pass
        finally:
            if user_id in user_tasks:
                del user_tasks[user_id]

# 多消息验证
async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ 请<b>等待</b>回复上一条消息\n"
        text += "或者你可以/cancel取消它"
        await update.message.chat.send_action(action=ChatAction.TYPING)
        await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        return True
    else:
        return False

# 解释图片
async def photo_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    prompt = update.message.caption or "请问图片里有什么？"
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    current_model = db.get_user_attribute(user_id, "current_model")
    if "vision" not in current_model:
        text = "🥲 当前模型不支持发送图片."
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        return

    photo = update.message.photo
    photo_file = await context.bot.get_file(photo[-1].file_id)
    temp_file = io.BytesIO(await photo_file.download_as_bytearray())
    temp_file_png = io.BytesIO()
    original_image = Image.open(temp_file)
    original_image.save(temp_file_png, format='PNG')
    
    content = [{'type':'text', 'text': prompt}, {'type':'image_url', \
                    'image_url': {'url':openai_utils.encode_image(temp_file_png), 'detail':config.vision_detail } }]
    
    await message_handle(update, context, message=content)

# 语音转文字
async def voice_message_handle(update: Update, context: CallbackContext):
    # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return

    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = update.message.voice
    voice_file = await context.bot.get_file(voice.file_id)
    
    # store file in memory, not on disk
    buf = io.BytesIO()
    await voice_file.download_to_memory(buf)
    buf.name = "voice.oga"  # file extension is required
    buf.seek(0)  # move cursor to the beginning of the buffer

    transcribed_text = await openai_utils.transcribe_audio(buf)
    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    # update n_transcribed_seconds
    db.set_user_attribute(user_id, "n_transcribed_seconds", voice.duration + db.get_user_attribute(user_id, "n_transcribed_seconds"))

    await message_handle(update, context, message=transcribed_text)

# 文字转语音
async def generate_audio_handle(update: Update, context: CallbackContext, message=None):
     # check if bot was mentioned (for group chats)
    if not await is_bot_mentioned(update, context):
        return
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    message = message or update.message.text.replace("/audio","").strip()
    if message is None or len(message) == 0:
        await update.message.reply_text("🥲 请输入/audio <b>需要生成语音的文字内容</b>。 请再试一次！", parse_mode=ParseMode.HTML)
        return
    current_model = db.get_user_attribute(user_id, "current_audio_model")
    audio_file = await openai_utils.generate_audio(message, current_model)
     # token usage
    db.set_user_attribute(user_id, current_model, db.get_user_attribute(user_id, current_model) + len(message))

    await update.message.chat.send_action(action=ChatAction.UPLOAD_VOICE)
    await update.effective_message.reply_voice(
        reply_to_message_id=update.message.message_id,
        voice=audio_file
    )
    audio_file.close()

# 图片生成
async def generate_image_handle(update: Update, context: CallbackContext, message=None):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    message = message or update.message.text.replace("/img","").strip()

    if message is None or len(message) == 0:
        await update.message.reply_text("🥲 请输入/img <b>需要生成的图片描述内容</b>。 请再试一次！", parse_mode=ParseMode.HTML)
        return
    current_model = db.get_user_attribute(user_id, "current_image_model")
    try:
        image_urls = await openai_utils.generate_images(message, current_model)
    except Exception as e:
        if str(e).startswith("Your request was rejected as a result of our safety system"):
            text = "🥲 您的请求<b>不符合</b> OpenAI 的使用政策。"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            return
        else:
            raise
    # token usage
    db.set_user_attribute(user_id, current_model, config.return_n_generated_images + db.get_user_attribute(user_id, current_model))

    for i, image_url in enumerate(image_urls):
        await update.message.chat.send_action(action=ChatAction.UPLOAD_PHOTO)
        await update.message.reply_photo(image_url, parse_mode=ParseMode.HTML)

# 开始新对话
async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    db.start_new_dialog(user_id)
    await update.message.chat.send_action(action=ChatAction.TYPING)
    await update.message.reply_text("开始新对话 ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{config.chat_modes[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)

# 取消对话任务
async def cancel_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    if user_id in user_tasks:
        task = user_tasks[user_id]
        task.cancel()
    else:
        await update.message.chat.send_action(action=ChatAction.TYPING)
        await update.message.reply_text("<i>没有什么可以取消...</i>", parse_mode=ParseMode.HTML)

# 获取角色预设菜单
def get_chat_mode_menu(page_index: int):
    n_chat_modes_per_page = config.n_chat_modes_per_page
    text = f"选择 <b>聊天模式</b> ({len(config.chat_modes)} 可用模式):"

    # buttons
    chat_mode_keys = list(config.chat_modes.keys())
    page_chat_mode_keys = chat_mode_keys[page_index * n_chat_modes_per_page:(page_index + 1) * n_chat_modes_per_page]

    keyboard = []
    for chat_mode_key in page_chat_mode_keys:
        name = config.chat_modes[chat_mode_key]["name"]
        keyboard.append([InlineKeyboardButton(name, callback_data=f"set_chat_mode|{chat_mode_key}")])

    # pagination
    if len(chat_mode_keys) > n_chat_modes_per_page:
        is_first_page = (page_index == 0)
        is_last_page = ((page_index + 1) * n_chat_modes_per_page >= len(chat_mode_keys))

        if is_first_page:
            keyboard.append([
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])
        elif is_last_page:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
            ])
        else:
            keyboard.append([
                InlineKeyboardButton("«", callback_data=f"show_chat_modes|{page_index - 1}"),
                InlineKeyboardButton("»", callback_data=f"show_chat_modes|{page_index + 1}")
            ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    return text, reply_markup

# 显示角色预设菜单
async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    text, reply_markup = get_chat_mode_menu(0)
    await update.message.chat.send_action(action=ChatAction.TYPING)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def show_chat_modes_callback_handle(update: Update, context: CallbackContext):
     await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
     if await is_previous_message_not_answered_yet(update.callback_query, context): return

     user_id = update.callback_query.from_user.id
     db.set_user_attribute(user_id, "last_interaction", datetime.now())

     query = update.callback_query
     await query.answer()

     page_index = int(query.data.split("|")[1])
     if page_index < 0:
         return

     text, reply_markup = get_chat_mode_menu(page_index)
     try:
         await update.message.chat.send_action(action=ChatAction.TYPING)
         await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
     except telegram.error.BadRequest as e:
         if str(e).startswith("Message is not modified"):
             pass

# 设置角色预设
async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)
    await update.message.chat.send_action(action=ChatAction.TYPING)
    await context.bot.send_message(
        update.callback_query.message.chat.id,
        f"{config.chat_modes[chat_mode]['welcome_message']}",
        parse_mode=ParseMode.HTML
    )

# 获取模型选择菜单
def get_model_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_model")
    text = "\n选择GPT模型:"
    # buttons to choose models
    buttons = []
    for model_key in config.models["available_text_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_model|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])
    return text, reply_markup

# 设置模型选择菜单
async def set_model_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_model", model_key)
    db.start_new_dialog(user_id)

    text, reply_markup = get_model_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass

def get_audio_model_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_audio_model")
    text = "\n选择语音模型:"
    # buttons to choose models
    buttons = []
    for model_key in config.models["available_audio_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_audio_model|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])
    return text, reply_markup

# 设置模型选择菜单
async def set_audio_model_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_audio_model", model_key)

    text, reply_markup = get_audio_model_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass

def get_image_model_menu(user_id: int):
    current_model = db.get_user_attribute(user_id, "current_image_model")
    text = "\n选择图像模型:"
    # buttons to choose models
    buttons = []
    for model_key in config.models["available_image_models"]:
        title = config.models["info"][model_key]["name"]
        if model_key == current_model:
            title = "✅ " + title

        buttons.append(
            InlineKeyboardButton(title, callback_data=f"set_image_model|{model_key}")
        )
    reply_markup = InlineKeyboardMarkup([buttons])
    return text, reply_markup

# 设置模型选择菜单
async def set_image_model_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    _, model_key = query.data.split("|")
    db.set_user_attribute(user_id, "current_image_model", model_key)

    text, reply_markup = get_image_model_menu(user_id)
    try:
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if str(e).startswith("Message is not modified"):
            pass

# 显示模型选择菜单
async def show_model_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await update.message.chat.send_action(action=ChatAction.TYPING)
    text, reply_markup = get_model_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

    text, reply_markup = get_image_model_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

    text, reply_markup = get_audio_model_menu(user_id)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

# 显示用量
async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    # count total usage statistics
    total_n_spent_dollars = 0
    total_n_used_tokens = 0

    n_used_tokens_dict = db.get_user_attribute(user_id, "n_used_tokens")
    n_transcribed_seconds = db.get_user_attribute(user_id, "n_transcribed_seconds")

    details_text = "🏷️ 详细信息:\n"
    for model_key in sorted(n_used_tokens_dict.keys()):
        n_input_tokens, n_output_tokens = n_used_tokens_dict[model_key]["n_input_tokens"], n_used_tokens_dict[model_key]["n_output_tokens"]
        total_n_used_tokens += n_input_tokens + n_output_tokens

        n_input_spent_dollars = config.models["info"][model_key]["price_per_1000_input_tokens"] * (n_input_tokens / 1000)
        n_output_spent_dollars = config.models["info"][model_key]["price_per_1000_output_tokens"] * (n_output_tokens / 1000)
        total_n_spent_dollars += n_input_spent_dollars + n_output_spent_dollars
        name = config.models["info"][model_key]["name"]
        details_text += f"- {name}: <b>{n_input_spent_dollars + n_output_spent_dollars:.03f}$</b> / <b>{n_input_tokens + n_output_tokens} tokens</b>\n"
    
    # image generation
    for model_key in config.models["available_image_models"]:
        n_generated_images = db.get_user_attribute(user_id, model_key)
        image_generation_n_spent_dollars = config.models["info"][model_key]["price_per_1_image"] * n_generated_images
        total_n_spent_dollars += image_generation_n_spent_dollars
        name = config.models["info"][model_key]["name"]
        details_text += f"- {name}: <b>{image_generation_n_spent_dollars:.03f}$</b> / <b>{n_generated_images} generated images</b>\n"

     # tts
    for model_key in config.models["available_audio_models"]:
        n_tts_used_tokens = db.get_user_attribute(user_id, model_key)
        tts_spent_dollars = config.models["info"][model_key]["price_per_1000_input_tokens"] * (n_tts_used_tokens / 1000)
        total_n_spent_dollars += tts_spent_dollars
        name = config.models["info"][model_key]["name"]
        details_text += f"- {name}: <b>{tts_spent_dollars:.03f}$</b> / <b>{n_tts_used_tokens} tokens</b>\n"

    # voice recognition
    voice_recognition_n_spent_dollars = config.models["info"]["whisper"]["price_per_1_min"] * (n_transcribed_seconds / 60)
    name = config.models["info"]["whisper"]["name"]
    details_text += f"- {name}: <b>{voice_recognition_n_spent_dollars:.03f}$</b> / <b>{n_transcribed_seconds:.01f} seconds</b>\n"
    total_n_spent_dollars += voice_recognition_n_spent_dollars
    
    text = f"你花了 <b>{total_n_spent_dollars:.03f}$</b>\n"
    text += f"你使用了 <b>{total_n_used_tokens}</b> tokens\n\n"
    text += details_text
    await update.message.chat.send_action(action=ChatAction.TYPING)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

# 重新编辑消息
async def edited_message_handle(update: Update, context: CallbackContext):
    if update.edited_message.chat.type == "private":
        text = "🥲 遗憾的是，不支持消息<b>编辑</b>"
        await update.message.chat.send_action(action=ChatAction.TYPING)
        await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)

# 错误处理
async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="处理更新时出现异常：", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"处理更新时引发异常\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )
        await update.message.chat.send_action(action=ChatAction.TYPING)
        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(update.effective_chat.id, "处理程序中出现一些错误")

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new", "开始新对话"),
        BotCommand("/mode", "选择角色预设"),
        BotCommand("/retry", "重新生成最后一个答案"),
        BotCommand("/img", "生成图片"),
        BotCommand("/audio", "生成语言"),
        BotCommand("/balance", "显示使用统计"),
        BotCommand("/model", "显示模型选择"),
        BotCommand("/cancel", "取消任务"),
        BotCommand("/help", "显示帮助"),
    ])

def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .http_version("1.1")
        .get_updates_http_version("1.1")
        .post_init(post_init)
        .build()
    )

    # add handlers
    user_filter = filters.ALL
    if len(config.allowed_telegram_usernames) > 0:
        usernames = [x for x in config.allowed_telegram_usernames if isinstance(x, str)]
        any_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int)]
        user_ids = [x for x in any_ids if x > 0]
        group_ids = [x for x in any_ids if x < 0]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids) | filters.Chat(chat_id=group_ids)

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))
    application.add_handler(MessageHandler((filters.PHOTO | filters.VIDEO) & user_filter, photo_message_handle))
    
    application.add_handler(CommandHandler("audio", generate_audio_handle, filters=user_filter))
    application.add_handler(CommandHandler("img", generate_image_handle, filters=user_filter))
    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))
    application.add_handler(CommandHandler("help_group_chat", help_group_chat_handle, filters=user_filter))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))
    application.add_handler(CommandHandler("cancel", cancel_handle, filters=user_filter))
    application.add_handler(CommandHandler("model", show_model_handle, filters=user_filter))
    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CommandHandler("balance", show_balance_handle, filters=user_filter))

    application.add_handler(CallbackQueryHandler(show_chat_modes_callback_handle, pattern="^show_chat_modes"))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))
    application.add_handler(CallbackQueryHandler(set_model_handle, pattern="^set_model"))
    application.add_handler(CallbackQueryHandler(set_image_model_handle, pattern="^set_image_model"))
    application.add_handler(CallbackQueryHandler(set_audio_model_handle, pattern="^set_audio_model"))

    application.add_error_handler(error_handle)
    # start the bot
    application.run_polling()


if __name__ == "__main__":
    run_bot()
