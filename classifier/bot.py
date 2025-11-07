from io import BytesIO
import requests
from PIL import Image
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters

from bot_token import TOKEN  # must define: TOKEN = "123:ABC"
from model import inference, warmup  # must be: def inference(img1_pil, img2_pil) -> int

INCORRECT = "Incorrect input."
IMPOSSIBLE = "Impossible to load images."
DIFF = "These are photos of different rooms."
SAME = "These are photos of the same room."
ERR = "There was an error in the decision process."
WAIT = "Operation in process."

def _load_image(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        ct = r.headers.get("Content-Type", "")
        if "image" not in ct.lower() and not url.lower().endswith((".jpg")):
            return None
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def _parse(text: str):
    if "|" not in text:
        return None, None
    left, right = [part.strip() for part in text.split("|", 1)]
    if not (left and right and left.startswith(("http://", "https://")) and right.startswith(("http://", "https://"))):
        return None, None
    return left, right

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not msg.text:
        await msg.reply_text(INCORRECT)
        return

    url1, url2 = _parse(msg.text)
    if not url1 or not url2:
        await msg.reply_text(INCORRECT)
        return

    img1 = _load_image(url1)
    img2 = _load_image(url2)
    if img1 is None or img2 is None:
        await msg.reply_text(IMPOSSIBLE)
        return

    await msg.reply_text(WAIT)
    try:
        res = inference(img1, img2)
    except Exception:
        await msg.reply_text(ERR)
        return

    if res == 0:
        await msg.reply_text(DIFF)
    elif res == 1:
        await msg.reply_text(SAME)
    else:
        await msg.reply_text(ERR)

def main():
    warmup() 
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()