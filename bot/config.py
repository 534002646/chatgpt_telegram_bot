import yaml
import dotenv
from pathlib import Path

default_model = "gpt-4-1106-preview"
default_audio_model = "tts-1"
default_image_model = "dall-e-2"

config_dir = Path(__file__).parent.parent.resolve() / "config"

# load yaml config
with open(config_dir / "config.yml", 'r') as f:
    config_yaml = yaml.safe_load(f)

# load .env config
config_env = dotenv.dotenv_values(config_dir / "config.env")

# config parameters
telegram_token = config_yaml["telegram_token"]
openai_api_key = config_yaml["openai_api_key"]
openai_api_base = config_yaml.get("openai_api_base", None)
allowed_telegram_usernames = config_yaml["allowed_telegram_usernames"]
new_dialog_timeout = config_yaml["new_dialog_timeout"]
return_n_generated_images = config_yaml.get("return_n_generated_images", 1)

image_quality = config_yaml.get("image_quality", "standard")
image_style = config_yaml.get("image_style", "vivid")
image_size = config_yaml.get("image_size", "512x512")

tts_voice = config_yaml.get("tts_voice", "alloy")
vision_detail = config_yaml.get("vision_detail", "auto")
n_chat_modes_per_page = config_yaml.get("n_chat_modes_per_page", 1)
mongodb_uri = config_env['MONGODB_URI']
context_len = config_yaml.get("context_len", 5)

# chat_modes
with open(config_dir / "chat_modes.yml", 'r') as f:
    chat_modes = yaml.safe_load(f)

# models
with open(config_dir / "models.yml", 'r') as f:
    models = yaml.safe_load(f)

# files
help_group_chat_video_path = Path(__file__).parent.parent.resolve() / "static" / "help_group_chat.mp4"
