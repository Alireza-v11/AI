# 🧠 Brain Tumor Classification Using CNN

این پروژه شامل آموزش یک مدل شبکه عصبی کانولوشنال (CNN) برای تشخیص و طبقه‌بندی تومورهای مغزی از تصاویر MRI سیاه و سفید است.

---

## 📝 توضیح پروژه

هدف این پروژه، تشخیص چهار کلاس مختلف از تصاویر MRI مغز است:

- 🧩 **Pituitary tumor**  
- 🧠 **Meningioma tumor**  
- ⚡ **Glioma tumor**  
- ✅ **No tumor**  

مدل بر پایه CNN ساخته شده و تصاویر در اندازه 128x128 پردازش می‌شوند. داده‌های آموزش و تست در پوشه‌های جداگانه قرار دارند.

---

## 📂 ساختار پوشه‌ها

- `tumor/training` : شامل تصاویر آموزش با چهار زیرپوشه (pituitary, meningioma, glioma, notumor)  
- `tumor/testing` : شامل تصاویر تست با ساختار مشابه  
- `brain_tumor_cnn_model.h5` : فایل مدل آموزش دیده ذخیره شده  
- کدهای آموزش و تست مدل (مثلاً `train_model.py` و `test_model.py`)  

---

## ⚙️ توضیح کد

- استفاده از ImageDataGenerator برای بارگذاری و نرمال‌سازی تصاویر همراه با جداسازی داده‌های اعتبارسنجی از داده‌های آموزش (10٪).
- مدل شامل دو لایه کانولوشنال، دو لایه ماکس پولینگ، لایه Flatten، لایه Dense با 128 نرون و لایه خروجی Softmax برای چهار کلاس.
- آموزش مدل با 10 اپوک و بهینه‌ساز Adam.
- ذخیره مدل آموزش دیده در قالب فایل H5.
- ارزیابی مدل روی داده‌های تست و نمایش دقت.

---

## ⚠️ نکات مهم

- تصاویر باید تک کاناله (grayscale) باشند.  
- ابعاد تصاویر باید 128x128 باشد (در کد به صورت خودکار تغییر اندازه داده می‌شود).  

---

## 🔗 لینک‌ها

- 📁 دیتاست مورد استفاده در این پروژه (Kaggle):  
  [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- 💾 دانلود مدل آموزش دیده (Google Drive):  
  [brain_tumor_cnn_model.h5 - Google Drive](https://drive.google.com/file/d/1BlpPzOMWCBFfy8khFtYKbvbzfTvyTIOF/view?usp=sharing)
