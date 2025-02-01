from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from voice_cloner import VoiceCloner
import shutil
import os
from pathlib import Path
from typing import List
from pydantic import BaseModel
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy import signal

app = FastAPI(title="Ses Klonlama API")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dosyalar ve template'ler için yapılandırma
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
templates = Jinja2Templates(directory="templates")

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('static/favicon.ico')

# Klasör yapılandırmaları
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("output")
INPUT_DIR = Path("input")

# Klasörleri oluştur
for directory in [TEMP_DIR, OUTPUT_DIR, INPUT_DIR]:
    directory.mkdir(exist_ok=True)

# Global değişkenler
voice_cloner = None
selected_reference_file = None
_preloaded_model = None
_model_loading = False
_model_ready = asyncio.Event()

@app.on_event("startup")
async def startup_event():
    """Uygulama başladığında modeli önceden yükle"""
    global _preloaded_model, _model_loading
    try:
        if not _model_loading:
            _model_loading = True
            print("Model önceden yükleniyor...")
            
            # Geçici bir referans dosyası yolu oluştur
            temp_ref = str(INPUT_DIR / "temp_ref.wav")
            
            # Eğer input klasöründe herhangi bir ses dosyası varsa, onu kullan
            for file in INPUT_DIR.iterdir():
                if file.suffix.lower() in {'.mp3', '.wav'}:
                    temp_ref = str(file)
                    break
            
            # Modeli ayrı bir thread'de yükle
            with ThreadPoolExecutor() as executor:
                future = executor.submit(VoiceCloner, temp_ref)
                _preloaded_model = future.result()
            
            print("Model başarıyla yüklendi ve GPU'ya aktarıldı!")
            _model_ready.set()
            
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        _preloaded_model = None
        _model_ready.set()
    finally:
        _model_loading = False

async def get_voice_cloner(reference_path: str) -> VoiceCloner:
    """Global voice cloner instance'ı al veya oluştur"""
    global voice_cloner
    
    # Model yüklenene kadar bekle
    await _model_ready.wait()
    
    if voice_cloner is None and _preloaded_model is not None: # Eğer aktif model yok ama önceden yüklenmiş model varsa
        # Önceden yüklenmiş modeli kullan
        voice_cloner = _preloaded_model
        voice_cloner._reference_audio_path = reference_path
    elif voice_cloner is None: # Eğer aktif model yoksa
        # Yeni model oluştur
        with ThreadPoolExecutor() as executor:
            future = executor.submit(VoiceCloner, reference_path)
            voice_cloner = future.result()
    else: # Eğer aktif model varsa
        # Mevcut modelin referans dosyasını güncelle
        voice_cloner._reference_audio_path = reference_path
    
    return voice_cloner

# Desteklenen ses dosyası uzantıları
ALLOWED_EXTENSIONS = {'.mp3', '.wav'}

@app.get("/reference-voices", response_model=List[str])
async def list_reference_voices():
    """Input klasöründeki mevcut ses dosyalarını listele"""
    voices = []
    for file in INPUT_DIR.iterdir():
        if file.suffix.lower() in ALLOWED_EXTENSIONS:
            voices.append(file.name)
    return voices

def process_audio(audio_path: str) -> str:
    """
    Ses dosyasını profesyonel şekilde işler.
    
    İşlemler:
    1. Gürültü azaltma
    2. Dinamik aralık sıkıştırma (compression)
    3. Ses seviyesi normalizasyonu
    4. Frekans dengeleme (EQ)
    5. De-essing (tiz seslerdeki patlamaları azaltma)
    6. Son normalizasyon
    
    Args:
        audio_path: İşlenecek ses dosyasının yolu
        
    Returns:
        İşlenmiş ses dosyasının yolu
    """
    try:
        print("\n Ses işleme başlatılıyor...")
        print(f" İşlenecek dosya: {Path(audio_path).name}")
        
        # Ses dosyasını yükle
        print(" Ses dosyası yükleniyor...")
        y, sr = librosa.load(audio_path, sr=None)
        # y: ses sinyali (numpy array)
        # sr: ses oranı (samples per second)
        # sr=None: orijinal örnekleme hızını korur

        print(f" Örnek oranı: {sr} Hz")
        print(f" Ses uzunluğu: {len(y)/sr:.2f} saniye")
        
        # DC offset'i kaldır
        """
        DC Offset, ses sinyalinin sıfır çizgisinden (orta nokta) yukarı veya aşağı kaymasıdır.
        Bu durum genellikle:
            -Ses kayıt cihazlarındaki elektronik problemlerden
            -Düşük kaliteli mikrofonlardan
            -Ses kartı sorunlarından kaynaklanabilir
        Faydaları:
            -Daha tutarlı ses seviyesi
            -Daha iyi ses kalitesi
            -Sonraki işlemler için daha uygun sinyal
        """
        print(" DC offset düzeltiliyor...")
        y = librosa.util.normalize(y - np.mean(y))
        
        # Gürültü azaltma
        print(" Gürültü azaltma uygulanıyor...")
        y_reduced = nr.reduce_noise(
            y=y,
            sr=sr,
            prop_decrease=0.85, # Gürültü azaltma oranını
            stationary=True, # Gürültü tipi
            n_jobs=1
        )
        
        # Dinamik aralık sıkıştırma
        # threshold: Sıkıştırma eşiği (dB cinsinden)
        # ratio: Sıkıştırma oranı
        print(" Dinamik aralık sıkıştırması uygulanıyor...")
        def compress_dynamic_range(y, threshold=-20, ratio=2.5):
            db = 20 * np.log10(np.abs(y) + 1e-8) # Sinyalin eşiğini dB cinsinden hesapla
            mask = db > threshold 
            db[mask] = threshold + (db[mask] - threshold) / ratio
            return np.sign(y) * (10 ** (db / 20))
        
        y_compressed = compress_dynamic_range(y_reduced)
        
        print(" Frekans dengeleme (EQ) uygulanıyor...")
        def apply_eq(y, sr):
            # Bas frekansları güçlendir (100-200 Hz)
            bass_freq = 150 # Merkez frekans
            bass_width = 100 # Bant genişliği
            bass_gain = 1.2 # x1.2 güçlendirme oranı
            
            # Orta frekansları düzenle (1000-2000 Hz)
            mid_freq = 1500 # Merkez frekans
            mid_width = 1000 # Bant genişliği
            mid_gain = 1.1 # x1.1 güçlendirme oranı
            
            # Tiz frekansları kontrol et (5000-7000 Hz)
            treble_freq = 6000 # Merkez frekans
            treble_width = 2000 # Bant genişliği
            treble_gain = 0.95 # x0.95 güçlendirme oranı
            
            # FFT uygula
            # Her frekans bileşenini ayrı ayrı işlemeye olanak sağlar.
            D = librosa.stft(y) # Short-Time Fourier Transform (STFT)
            
            
            # Frekans bantlarını belirle
            # Her frekans bandının Hz cinsinden değerini hesaplar
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Bas frekansları güçlendir
            bass_mask = np.logical_and(freqs >= bass_freq - bass_width/2,
                                     freqs <= bass_freq + bass_width/2)
            D[bass_mask] *= bass_gain
            
            # Orta frekansları düzenle
            mid_mask = np.logical_and(freqs >= mid_freq - mid_width/2,
                                    freqs <= mid_freq + mid_width/2)
            D[mid_mask] *= mid_gain
            
            # Tiz frekansları kontrol et
            treble_mask = np.logical_and(freqs >= treble_freq - treble_width/2,
                                       freqs <= treble_freq + treble_width/2)
            D[treble_mask] *= treble_gain
            
            # Ters FFT ile zaman domenine dön
            return librosa.istft(D)
        
        y_eq = apply_eq(y_compressed, sr)
        
        # De-essing
        print(" De-essing uygulanıyor...")
        def deess(y, sr):
            cutoff = 5000
            b, a = signal.butter(4, cutoff/(sr/2), btype='highpass')
            high_freqs = signal.filtfilt(b, a, y)
            
            envelope = np.abs(high_freqs)
            threshold = np.percentile(envelope, 95)
            mask = envelope > threshold
            
            smoothing_window = int(0.01 * sr)
            smoothing_kernel = np.hanning(smoothing_window)
            smoothing_kernel /= smoothing_kernel.sum()
            smooth_mask = np.convolve(mask.astype(float), smoothing_kernel, mode='same')
            
            reduction_factor = 0.5
            gain = 1 - (reduction_factor * smooth_mask)
            return y * gain
            
        y_deessed = deess(y_eq, sr)
        
        # Son normalizasyon
        print(" Son normalizasyon ve limitleme uygulanıyor...")
        def normalize_with_limiter(y, target_lufs=-14):
            rms = np.sqrt(np.mean(y**2))
            target_rms = 10 ** (target_lufs/20)
            gain = target_rms / rms
            
            y_normalized = np.tanh(gain * y)
            
            max_peak = 0.99
            if np.max(np.abs(y_normalized)) > max_peak:
                y_normalized = y_normalized * (max_peak / np.max(np.abs(y_normalized)))
            
            return y_normalized
            
        y_final = normalize_with_limiter(y_deessed)
        
        # İşlenmiş dosyayı kaydet
        print(" İşlenmiş ses dosyası kaydediliyor...")
        processed_path = str(Path(audio_path).with_suffix('')) + '_processed' + Path(audio_path).suffix
        sf.write(processed_path, y_final, sr)
        
        print(" Ses işleme başarıyla tamamlandı!")
        return processed_path
        
    except Exception as e:
        print(f" Ses işleme hatası: {str(e)}")
        return audio_path

@app.post("/upload-reference")
async def upload_reference(file: UploadFile = File(...)):
    """Yeni referans ses dosyası yükle"""
    try:
        print("\n Yeni ses dosyası yükleniyor...")
        print(f" Dosya adı: {file.filename}")
        
        # Dosya uzantısını kontrol et
        if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            error_msg = f" Hata: Sadece {', '.join(ALLOWED_EXTENSIONS)} uzantılı dosyalar kabul edilir"
            print(error_msg)
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        
        # Dosyayı kaydet
        file_path = INPUT_DIR / file.filename
        print(f" Dosya kaydediliyor: {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(" Dosya başarıyla kaydedildi")
        print(" Ses işleme başlatılıyor...")
            
        # Ses dosyasını işle
        processed_path = process_audio(str(file_path))
        
        return {"filename": file.filename, "status": "success"}
        
    except Exception as e:
        error_msg = f" Hata: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/select-reference")
async def select_reference(filename: str = Form(..., description="Referans ses dosyasının adı")):
    """Input klasöründen referans ses seç"""
    global selected_reference_file
    
    try:
        file_path = INPUT_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Dosya bulunamadı: {filename}")
        
        if not filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            raise HTTPException(status_code=400, detail="Geçersiz dosya formatı")
        
        selected_reference_file = str(file_path)
        
        # Voice cloner'ı hazırla
        await get_voice_cloner(selected_reference_file)
        
        return {"message": f"Referans ses seçildi: {filename}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BatchTextRequest(BaseModel):
    texts: List[str]

@app.post("/clone-voice-batch")
async def clone_voice_batch(request: BatchTextRequest):
    """Batch processing için endpoint"""
    if not selected_reference_file:
        raise HTTPException(status_code=400, detail="Önce referans ses seçin!")
    
    try:
        vc = await get_voice_cloner(selected_reference_file)
        
        # Benzersiz bir dosya adı oluştur
        timestamp = int(time.time())
        base_output = f"output_batch_{timestamp}"
        
        # Batch işlemi başlat
        output_files = vc.clone_voice_batch(request.texts, base_output)
        
        if not output_files:
            raise HTTPException(status_code=500, detail="Ses oluşturma başarısız!")
        
        return {"message": "Batch işlemi tamamlandı", "files": output_files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clone-voice")
async def clone_voice(
    filename: str = Form(None),
    text: str = Form(None),
    filename_query: str = Query(None, alias="filename"),
    text_query: str = Query(None, alias="text")
):
    """Clone voice using specified input file and text"""
    # Form veya query parametrelerinden değerleri al
    filename = filename or filename_query
    text = text or text_query
    
    if not filename or not text:
        raise HTTPException(status_code=400, detail="Dosya adı ve metin gerekli!")
    
    try:
        # Referans ses dosyası yolunu oluştur
        reference_path = str(INPUT_DIR / filename)
        if not os.path.exists(reference_path):
            raise HTTPException(status_code=404, detail=f"Dosya bulunamadı: {filename}")
        
        # Voice cloner'ı hazırla
        vc = await get_voice_cloner(reference_path)
        
        # Benzersiz bir çıktı dosyası adı oluştur
        timestamp = int(time.time())
        output_filename = f"output_{timestamp}.wav"
        output_path = str(OUTPUT_DIR / output_filename)
        
        # Ses oluştur
        result = vc.clone_voice(text, output_path)
        
        if not result:
            raise HTTPException(status_code=500, detail="Ses oluşturma başarısız!")
        
        # Başarılı sonuç döndür
        return FileResponse(
            result,
            media_type="audio/wav",
            filename=output_filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LongTextRequest(BaseModel):
    text: str
    filename: str = None

class LongTextResponse(BaseModel):
    file_path: str
    progress: dict = None

@app.post("/clone-voice-long", response_model=LongTextResponse)
async def clone_voice_long(request: Request, text_request: LongTextRequest, subject_id: str = Query(None)):
    """Uzun metinleri işleyerek ses dosyası oluşturur"""
    try:
        if not text_request.text:
            raise HTTPException(status_code=400, detail="Text is required")
            
        # Referans ses dosyası kontrolü
        if not selected_reference_file:
            raise HTTPException(status_code=400, detail="No reference voice selected")
            
        # Voice cloner instance'ı al
        vc = await get_voice_cloner(selected_reference_file)
        
        # Output dosya adını belirle
        filename = text_request.filename or f"long_output_{int(time.time())}.wav"
        output_path = str(OUTPUT_DIR / filename)
        
        progress_data = {}
        
        def update_progress(progress):
            nonlocal progress_data
            progress_data = progress
        
        # Ses dönüşümünü gerçekleştir
        file_path = vc.clone_voice_long_text(text_request.text, output_path, progress_callback=update_progress)
        
        if not file_path:
            raise HTTPException(status_code=500, detail="Voice cloning failed")
            
        # Dosya yolunu /output/ ile başlayacak şekilde düzenle
        relative_path = f"/output/{os.path.basename(file_path)}"

        try:
            # MongoDB'ye kaydet
            auth_header = request.headers.get('Authorization')
            if auth_header and subject_id:
                response = requests.post(
                    'http://localhost:3000/api/audio',
                    headers={
                        'Authorization': auth_header,
                        'Content-Type': 'application/json'
                    },
                    json={
                        'subjectId': subject_id,
                        'audioPath': relative_path,
                        'text': text_request.text
                    },
                    timeout=10  # 10 saniye timeout
                )
                
                if response.status_code == 201:
                    print(f"Audio saved to MongoDB successfully: {response.json()}")
                else:
                    print(f"Failed to save audio to MongoDB. Status: {response.status_code}, Response: {response.text}")
            else:
                print("Missing Authorization header or subjectId")
        except Exception as e:
            print(f"Error saving to MongoDB: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return LongTextResponse(
            file_path=relative_path,
            progress=progress_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Ana sayfa"""
    voices = await list_reference_voices()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "voices": voices}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)