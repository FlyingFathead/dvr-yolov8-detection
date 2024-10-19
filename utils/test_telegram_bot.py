# test_telegram_bot.py
# (to test that your bot is up and running if in use)

import os
import telegram
import asyncio

BOT_TOKEN = os.getenv('DVR_YOLOV8_TELEGRAM_BOT_TOKEN')
ALLOWED_USERS = os.getenv('DVR_YOLOV8_ALLOWED_TELEGRAM_USERS')

async def send_test_message():
    if BOT_TOKEN and ALLOWED_USERS:
        bot = telegram.Bot(token=BOT_TOKEN)
        allowed_users = [int(uid.strip()) for uid in ALLOWED_USERS.split(',')]
        
        for user_id in allowed_users:
            try:
                print(f"Sending test message to user {user_id}...")
                await bot.send_message(chat_id=user_id, text="Test message from your YOLOv8 detection system.")
                print(f"Message sent to user {user_id}.")
            except Exception as e:
                print(f"Failed to send message to user {user_id}: {e}")
    else:
        print("Bot token or allowed users not set.")

# Run the async function
if __name__ == "__main__":
    asyncio.run(send_test_message())
