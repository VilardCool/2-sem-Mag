from telegram.ext import *
import tensorflow as tf
from skimage.io import imread, imsave
import numpy as np
from config import TOKEN

PHOTO, COMPRESSION = range(2)
photo_name = "img.png"
res_name = "res.png"
img = []

model = tf.keras.models.load_model("model.keras")

async def start(update, context):
    user = update.effective_user

async def image(update, context):
    await update.message.reply_text("Send image")
    return PHOTO

async def compression(update, context):
    await update.message.reply_text("Wait for result")

    img = imread(photo_name)

    test = [(img/255.0).tolist(), (img/255.0).tolist()]

    decoded_imgs = model.predict(test)
    res_img = (decoded_imgs[0]* 255).astype(np.uint8)
    imsave(res_name, res_img)

    await update.message.reply_text("Result:")
    await context.bot.send_photo(chat_id=update.message.chat_id, photo=open(res_name, 'rb'))

    return ConversationHandler.END

async def photo(update, context):
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive(photo_name)
    await update.message.reply_text("Send any text")
    return COMPRESSION

async def cancel(update, context):
    await update.message.reply_text("Operation canceled")
    return ConversationHandler.END

def main() -> None:
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))

    uq_handler = ConversationHandler(
        entry_points=[CommandHandler("image", image)],
        states={
            PHOTO: [MessageHandler(filters.ALL, photo)],
            COMPRESSION: [MessageHandler(filters.ALL, compression)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    application.add_handler(uq_handler)

    application.run_polling()


if __name__ == "__main__":
    main()