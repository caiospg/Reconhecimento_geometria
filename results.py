# predict_final.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from pathlib import Path

IMG_SIZE = (128,128)
MODEL_PATH = "mobilenet_shapes_final.keras"

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ["circle","square","triangle"]

def predict_shape(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    return class_names[idx], float(preds[idx]), dict(zip(class_names, preds))

def show_prediction(img_path, pred_class, pred_prob, all_probs):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{pred_class.upper()} ({pred_prob*100:.1f}%)", fontsize=14, color='green')
    plt.subplot(1,2,2)
    plt.bar(all_probs.keys(), all_probs.values(), color='skyblue')
    plt.ylim(0,1)
    plt.title("Probabilidades das classes")
    plt.ylabel("Confiança")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Reconhecimento de Formas Geométricas ===")
    while True:
        path = input("\nCaminho da imagem (ou 'sair'): ").strip()
        if path.lower() == "sair":
            break
        img_file = Path(path)
        if not img_file.exists():
            print(f"Arquivo não encontrado: {path}")
            continue
        try:
            pred_class, pred_prob, all_probs = predict_shape(path)
            print(f"\nResultado: {pred_class.upper()} (confiança {pred_prob*100:.2f}%)")
            print("Probabilidades detalhadas:")
            for cls, prob in all_probs.items():
                print(f" - {cls}: {prob*100:.2f}%")
            show_prediction(path, pred_class, pred_prob, all_probs)
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
