import librosa
import tsfel
import pandas as pd
import numpy as np

def features_extraction(file, duration=30, hop_length=512):

    try:
        y, sr = librosa.load(file, sr=22050, mono=True, duration=duration)
        # обработка тихих учатков трека
        if y.size == 0:
            return None
        
        # mfcc - для тембра (высота и окраска)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # 7 основных нот + 5 дополнительных
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
        # частота, выше которой 15% энергии, высокий rolloff часто в драйвовых/агрессивных моментах
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        # "яркость": высокий - энергичность, низкий - спокойствие
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        # частота смены знака, высокий - шумный, низкий - спокойный
        zcr = librosa.feature.zero_crossing_rate(y)
        # средняя энергия, громкость звука
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        # ширина спектра (насыщенность/возбуждение)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        # шумность/гармоничность
        flatness = librosa.feature.spectral_flatness(y=y)
    
        # темп/ритм
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
        features_dict = {
    
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'rolloff_mean': float(np.mean(rolloff)),
            'zcr_mean': float(np.mean(zcr)),
            'rms_mean': float(np.mean(rms)),
            'bandwidth_mean': float(np.mean(bandwidth)),
            'flatness_mean': float(np.mean(flatness)),
    
            'spectral_centroid_std': float(np.std(spectral_centroid)),
            'rolloff_std': float(np.std(rolloff)),
            'zcr_std': float(np.std(zcr)), 
            'rms_std': float(np.std(rms)), 
            'bandwidth_std': float(np.std(bandwidth)), 
            'flatness_std': float(np.std(flatness)), 
            
            'tempo': float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0]),
        }
        
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features_dict.update({f'mfcc_mean_{i+1}': float(v) for i, v in enumerate(mfcc_mean)})
        features_dict.update({f'mfcc_std_{i+1}': float(v) for i, v in enumerate(mfcc_std)})
    
        # средняя активность каждой ноты
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        features_dict.update({f'chroma_mean_{i+1}': float(v) for i, v in enumerate(chroma_mean)})
        features_dict.update({f'chroma_std_{i+1}': float(v) for i, v in enumerate(chroma_std)})

        df_librosa = pd.DataFrame([features_dict])

        # sample frequency для rms = частота дискретизации / шаг окна (свернутый сигнал)
        fs_rms = sr / float(hop_length) # 43 в сек

        cfg = tsfel.get_features_by_domain()
        del cfg['spectral']
        del cfg['fractal']

        df_tsfel = tsfel.time_series_features_extractor(cfg, rms[0], fs=fs_rms)

        return pd.concat([df_librosa, df_tsfel], axis=1).reset_index(drop=True)

    except Exception as e:
        print(f'Ошибка при обработке файла: {str(e)}')
        return None