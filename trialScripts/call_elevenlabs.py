from elevenlabs import play
from elevenlabs.client import ElevenLabs

client = ElevenLabs(
  api_key="sk_c8ad69dda6d2c161800317563a65af4b656b711b8f11501a", # Defaults to ELEVEN_API_KEY
)

audio = client.generate(
  text="Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!",
  voice="Brian",
  model="eleven_multilingual_v2"
)
play(audio)
