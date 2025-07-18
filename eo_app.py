import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import time
import os

# --- Названия классов ---
CLASS_NAMES = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning',
               'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

CLASS_NAMES_dict = {
    'dew': 'Роса',
    'fogsmog': 'Туман',
    'frost': 'Изморозь',
    'glaze': 'Гололёд',
    'hail': 'Град',
    'lightning': 'Молния',
    'rain': 'Ливень, потоп',
    'rainbow': 'Радуга',
    'rime': 'Иней',
    'sandstorm': 'Песчаная буря',
    'snow': 'Снегопад'
}

# --- Преобразования ---
transform_resnet18 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_all_another = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Конфигурация моделей ---
model_configs = {
    "ResNet-18": {
        "arch": "resnet18",
        "weights": "models/model_resnet18_weights.pt",
        "transform": transform_resnet18,
        "url": None  # Локально в репозитории
    },
    "ResNet-50": {
        "arch": "resnet50",
        "weights": "models/model_resnet50_weights.pt",
        "transform": transform_all_another,
        "url": None
    },
    "ResNet-101": {
        "arch": "resnet101",
        "weights": "models/model_resnet101_weights.pt",
        "transform": transform_all_another,
        "url": "https://huggingface.co/EugeneOss/Weather_ResNet101/resolve/main/model_resnet101_weights.pt"
    },
    "ResNet-152": {
        "arch": "resnet152",
        "weights": "models/model_resnet152_weights.pt",
        "transform": transform_all_another,
        "url": "https://huggingface.co/EugeneOss/Weather_ResNet101/resolve/main/model_resnet152_weights.pt"
    }
}

# --- Загрузка модели ---
def load_model(arch, weight_path, download_url):
    if not os.path.exists(weight_path):
        if download_url is None:
            raise FileNotFoundError(f"Файл {weight_path} не найден и URL не указан.")
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        with open(weight_path, 'wb') as f:
            response = requests.get(download_url)
            response.raise_for_status()
            f.write(response.content)

    model_fn = {
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152
    }

    model = model_fn[arch](pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    return model

# --- Предсказание ---
def predict(model, image_tensor):
    with torch.no_grad():
        start_time = time.time()
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze()
        top_idx = torch.argmax(probs).item()
        end_time = time.time()
        duration = end_time - start_time
        return CLASS_NAMES[top_idx], probs[top_idx].item(), probs, duration

# --- Интерфейс ---
st.set_page_config(page_title="Weather Classification", layout="centered")
st.title("🌤️ Классификация погодных условий по фото")

# --- Выбор модели ---
model_choice = st.sidebar.radio("Выберите модель", list(model_configs.keys()))
config = model_configs[model_choice]
arch = config["arch"]
weights = config["weights"]
transform = config["transform"]
url = config["url"]

# --- Кэш модели ---
@st.cache_resource
def get_model():
    return load_model(arch, weights, url)

model = get_model()

# --- Способ загрузки изображения ---
option = st.radio("Способ загрузки изображения:", ["Загрузить файл", "Указать URL"], horizontal=True)
image = None

if option == "Загрузить файл":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Указать URL":
    url_input = st.text_input("Введите URL изображения:")
    if url_input:
        try:
            response = requests.get(url_input, timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")

# --- Вывод результата ---
if image is not None:
    st.image(image, caption="Загруженное изображение", use_container_width=True)
    image_tensor = transform(image).unsqueeze(0)
    pred_class, confidence, all_probs, duration = predict(model, image_tensor)

    st.success(f"**Предсказанный класс:** {CLASS_NAMES_dict[pred_class]}")
    st.info(f"**Уверенность:** {confidence:.4f}")
    st.metric("Время обработки", f"{duration:.3f} сек")

    with st.expander("📊 Показать вероятности по всем классам"):
        for cls, prob in zip(CLASS_NAMES, all_probs):
            st.write(f"{cls}: {prob:.4f}")
