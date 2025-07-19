from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import Update
from telegram.ext.callbackcontext import CallbackContext
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# مدل و کلاس‌ها
model = load_model("brain_tumor_cnn_model.h5")
class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
class_names_fa = {
    "glioma": "گلیوما",
    "meningioma": "منیژیوم",
    "no_tumor": "بدون تومور",
    "pituitary": "تومور هیپوفیز"
}

def start(update: Update, context: CallbackContext):
    welcome_msg = (
        "سلام! 👋\n"
        "من ربات تشخیص تومور مغزی هستم.\n"
        "لطفا یک تصویر MRI مغز خودتون رو برای من ارسال کنید تا مدل براتون تحلیل کنه."
    )
    update.message.reply_text(welcome_msg)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    return predicted_class, confidence

def handle_image(update: Update, context: CallbackContext):
    file = update.message.photo[-1].get_file()
    file_path = f"{file.file_id}.jpg"
    file.download(file_path)

    predicted_class, confidence = predict_image(file_path)
    os.remove(file_path)

    persian_name = class_names_fa.get(predicted_class, predicted_class)
    response = (
        f"🔍 تشخیص مدل:\n"
        f"نوع تومور (انگلیسی): *{predicted_class}*\n"
        f"نوع تومور (فارسی): *{persian_name}*\n"
        f"اعتماد مدل: {confidence:.2f}%"
    )
    update.message.reply_text(response, parse_mode="Markdown")

def main():
    TOKEN = "bot_token"
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_image))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
