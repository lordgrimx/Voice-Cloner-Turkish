from TTS.api import TTS
import os
from pathlib import Path
import shutil

def download_models():
    print("Coqui TTS modelini indiriliyor...")
    
    try:
        # Model dizinini yapılandır
        current_dir = os.getcwd()
        model_path = os.path.join(current_dir, "models")
        
        # Model dizinini oluştur
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # TTS için gerekli ortam değişkenlerini ayarla
        os.environ["COQUI_TOS_AGREED"] = "1"
        os.environ["TTS_HOME"] = model_path
        
        # Modelin tam yolu
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        expected_path = os.path.join(model_path, model_name.replace("/", "--"))
        
        # Eğer model zaten varsa, yeniden indirme
        if os.path.exists(expected_path):
            print(f"Model zaten mevcut: {expected_path}")
            return
            
        print(f"Model indirme konumu: {model_path}")
        tts = TTS(model_name=model_name)
        print(f"Model başarıyla indirildi! Konum: {expected_path}")
        
    except Exception as e:
        print(f"Model indirme hatası: {str(e)}")

if __name__ == "__main__":
    download_models()
