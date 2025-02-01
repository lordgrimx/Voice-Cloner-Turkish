import torch
from TTS.api import TTS
import os
from pathlib import Path
from typing import List, Optional
import time
import librosa
import soundfile as sf
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm

class VoiceCloner:
    def __init__(self, reference_audio_path):
        """Initialize voice cloner with reference audio"""
        if not isinstance(reference_audio_path, (str, bytes, os.PathLike)):
            raise TypeError("reference_audio_path must be a string or path!")
            
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {reference_audio_path}")
        
        # Temel ayarlar
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_path, "models", "tts", "tts_models--multilingual--multi-dataset--xtts_v2")
        self._reference_audio_path = reference_audio_path
        
        # CUDA optimizasyonları
        if self.device == "cuda":
            # GPU belleğini temizle
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # GPU bellek kontrolü - daha az model kullan
            available_memory = torch.cuda.get_device_properties(0).total_memory
            self.num_threads = int(max(1, available_memory // (2 * 1024 * 1024 * 1024)))  # Her model için daha fazla bellek ayır
            print(f"Using {self.num_threads} models based on GPU memory")
        else:
            self.num_threads = 1
        
        # Model havuzu
        self._model_pool = []
        self._model_locks = []  # Her model için ayrı kilit
        
        # Modelleri yükle
        print("\nLoading models...")
        for i in range(self.num_threads):
            try:
                # Her yüklemeden önce belleği temizle
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
                model = TTS(
                    model_path=self.model_path,
                    config_path=os.path.join(self.model_path, "config.json"),
                    progress_bar=False
                ).to(self.device)
                
                self._model_pool.append(model)
                self._model_locks.append(threading.Lock())
                print(f"Model {i} loaded successfully")
                
            except Exception as e:
                print(f"Error loading model {i}: {str(e)}")
                if i == 0:  # İlk model yüklenemezse hata ver
                    raise
        
        if not self._model_pool:
            raise RuntimeError("No models could be loaded!")
            
        print("\nModel pool ready!")
        
        # Output dizini
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into appropriate chunks"""
        try:
            # Noktalama işaretlerine göre böl
            sentences = re.split('[.!?]+', text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence = sentence + "."
                
                # Chunk boyutu kontrolü
                if len(current_chunk) + len(sentence) > 100:  # Daha küçük chunk'lar
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            print(f"Text chunking error: {str(e)}")
            return [text]

    def _process_chunk(self, args) -> Optional[str]:
        """Process a single chunk with a specific model"""
        chunk, chunk_idx, model_idx = args
        
        try:
            start_time = time.time()
            
            # Model ve kilit al
            model = self._model_pool[model_idx]
            lock = self._model_locks[model_idx]
            
            with lock:  # Her model için ayrı kilit kullan
                # Çıktı dosyası
                output_file = f"temp_output_{model_idx}_{chunk_idx}.wav"
                
                # Ses oluştur
                with torch.inference_mode():
                    model.tts_to_file(
                        text=chunk,
                        file_path=output_file,
                        speaker_wav=self._reference_audio_path,
                        language="tr"
                    )
                
                process_time = time.time() - start_time
                print(f" > Chunk {chunk_idx} (Model {model_idx}) - Time: {process_time:.2f}s")
                
                return output_file
            
        except Exception as e:
            print(f"Error processing chunk {chunk_idx} with model {model_idx}: {str(e)}")
            if self.device == "cuda":
                torch.cuda.empty_cache()  # Hata durumunda belleği temizle
            return None

    def clone_voice_long_text(self, text: str, output_path: str, progress_callback=None) -> Optional[str]:
        """Convert long text to speech using multiple models"""
        try:
            # Metni chunk'lara ayır
            chunks = self._chunk_text(text)
            total_chunks = len(chunks)
            print(f"\nProcessing {total_chunks} chunks")
            
            # İş listesi oluştur ve statik dağıt
            model_work_groups = [[] for _ in range(self.num_threads)]
            for i, chunk in enumerate(chunks):
                model_idx = i % self.num_threads
                model_work_groups[model_idx].append((chunk, i, model_idx))
            
            # Progress bar
            with tqdm(total=total_chunks, desc="Total Progress", position=0) as pbar:
                # Her model için ayrı bir progress bar
                model_pbars = [tqdm(total=len(group), desc=f"Model {i}", position=i+1, leave=True) 
                             for i, group in enumerate(model_work_groups)]
                
                # Thread havuzu ile işle
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    # Her model için ayrı bir işleyici fonksiyon
                    def process_group(model_idx):
                        results = []
                        for item in model_work_groups[model_idx]:
                            result = self._process_chunk(item)
                            if result:
                                results.append(result)
                                model_pbars[model_idx].update(1) # Model bazlı ilerleme
                                pbar.update(1) # Genel ilerleme
                        return results
                    
                    # Tüm modelleri aynı anda başlat
                    futures = []
                    for model_idx in range(self.num_threads):
                        if model_work_groups[model_idx]:
                            future = executor.submit(process_group, model_idx)
                            futures.append(future)
                    
                    # Sonuçları topla
                    output_files = []
                    for future in futures:
                        results = future.result()
                        if results:
                            output_files.extend(results)
                    
                    # Progress barları kapat
                    for pbar in model_pbars:
                        pbar.close()
            
            if not output_files:
                print("No audio files were generated!")
                return None
            
            # Sıralama için output dosyalarını index'e göre sırala
            output_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            # Ses dosyalarını birleştir
            print("\nCombining audio files...")
            self.combine_audio_files(output_files, output_path, progress_callback)
            
            # Geçici dosyaları temizle
            print("Cleaning up temporary files...")
            for file in output_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {file}: {str(e)}")
            
            return output_path
            
        except Exception as e:
            print(f"Error in long text processing: {str(e)}")
            return None

    def combine_audio_files(self, audio_files: List[str], output_path: str, progress_callback=None) -> str:
        """Combine multiple audio files into one"""
        if not audio_files:
            return None
            
        combined = None
        total_files = len(audio_files)
        
        for idx, file in enumerate(audio_files):
            if not os.path.exists(file):
                continue
                
            audio, sr = librosa.load(file, sr=None)
            
            if combined is None:
                combined = audio
            else:
                combined = np.concatenate([combined, audio])
            
            if progress_callback:
                progress = {
                    'current': idx + 1,
                    'total': total_files,
                    'percentage': ((idx + 1) / total_files) * 100
                }
                progress_callback(progress)
        
        if combined is not None:
            sf.write(output_path, combined, sr)
            return output_path  # Mutlak yol yerine göreceli yol döndür
        
        return None