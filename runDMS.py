import cv2
import numpy as np
import mediapipe as mp
import time
import os
import urllib.request
from collections import deque
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA

# --- CONFIGURAZIONE TIMER E SOGLIE ---
# Modifica questi valori per calibrare la sensibilità del sistema

# Soglie per il rilevamento del volto e degli occhi
EAR_THRESHOLD = 0.22           # Soglia per considerare l'occhio chiuso (tipicamente 0.20 - 0.25)
HEAD_POSE_THRESHOLD = 20.0     # Gradi massimi di pitch/yaw prima di considerare il conducente distratto

# Timer Sonno / Microsonno (in secondi)
TIMER_MICROSLEEP = 4.0         # Tempo di occhi chiusi per far scattare il "Microsonno"
TIMER_SLEEP = 7.0              # Tempo di occhi chiusi per far scattare il "Sonno"
TIMER_RESET_EYES = 1.0         # Tempo di occhi aperti consecutivo per resettare l'allarme sonno

# Timer Distrazione "Gufo" (in secondi)
TIMER_LONG_OWL = 5.0               # Tempo di distrazione continua per allarme "Distracted (long)"
TIMER_SHORT_OWL_CUMULATIVE = 10.0  # Tempo cumulativo di distrazione per allarme "Distracted (short)"...
TIMER_SHORT_OWL_WINDOW = 30.0      # ...calcolato all'interno di questa finestra temporale
TIMER_RESET_OWL = 2.0              # Tempo di attenzione continua per resettare l'allarme distrazione
# -------------------------------------

def download_model_if_needed():
    """Scarica il modello visivo di MediaPipe se non è presente localmente."""
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"Modello MediaPipe '{model_path}' non trovato.")
        print("Download in corso (circa 3MB)... attendere prego.")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download completato con successo!")

def get_head_pose(face_landmarks, img_w, img_h):
    """ Calcola l'orientamento della testa (pitch, yaw, roll) """
    face3d = np.array([
        (0.0, 0.0, 0.0),            # Punta del naso (1)
        (0.0, -330.0, -65.0),       # Mento (152)
        (-225.0, 170.0, -135.0),    # Angolo occhio sx (33)
        (225.0, 170.0, -135.0),     # Angolo occhio dx (263)
        (-150.0, -150.0, -125.0),   # Angolo bocca sx (61)
        (150.0, -150.0, -125.0)     # Angolo bocca dx (291)
    ], dtype=np.float64)

    face2d = np.array([
        (face_landmarks[1].x * img_w, face_landmarks[1].y * img_h),
        (face_landmarks[152].x * img_w, face_landmarks[152].y * img_h),
        (face_landmarks[33].x * img_w, face_landmarks[33].y * img_h),
        (face_landmarks[263].x * img_w, face_landmarks[263].y * img_h),
        (face_landmarks[61].x * img_w, face_landmarks[61].y * img_h),
        (face_landmarks[291].x * img_w, face_landmarks[291].y * img_h)
    ], dtype=np.float64)

    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                           [0, focal_length, img_h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face3d, face2d, cam_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))

    pitch, yaw, roll = angles[0][0], angles[1][0], angles[2][0]
    return pitch, yaw, roll

def get_ear(face_landmarks, img_w, img_h):
    """
    Calcola l'Eye Aspect Ratio (EAR) usando le posizioni delle palpebre.
    """
    def dist(p1_idx, p2_idx):
        x1, y1 = face_landmarks[p1_idx].x * img_w, face_landmarks[p1_idx].y * img_h
        x2, y2 = face_landmarks[p2_idx].x * img_w, face_landmarks[p2_idx].y * img_h
        return np.hypot(x1 - x2, y1 - y2)

    ear_left = (dist(160, 144) + dist(158, 153)) / (2.0 * dist(33, 133))
    ear_right = (dist(385, 380) + dist(387, 373)) / (2.0 * dist(362, 263))
    
    return (ear_left + ear_right) / 2.0

class HeartRateEstimator:
    def __init__(self, buffer_size=150, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.rgb_signals = []
        self.times = []
        self.ica = FastICA(n_components=3, random_state=0)
        
    def add_frame(self, image, face_landmarks, current_time, img_w, img_h):
        # Punto 10: centro fronte (superiore al naso) come Region Of Interest (ROI)
        x = int(face_landmarks[10].x * img_w)
        y = int(face_landmarks[10].y * img_h)
        
        # Estrarre un piccolo quadrato di pixel 20x20 per la ROI (sulla fronte)
        box_size = 10
        y1, y2 = max(0, y - box_size), min(img_h, y + box_size)
        x1, x2 = max(0, x - box_size), min(img_w, x + box_size)
        
        if y2 > y1 and x2 > x1:
            roi = image[y1:y2, x1:x2]
            avg_color = np.mean(roi, axis=(0, 1)) # BGR
            self.rgb_signals.append(avg_color)
            self.times.append(current_time)
            
        if len(self.rgb_signals) > self.buffer_size:
            self.rgb_signals.pop(0)
            self.times.pop(0)

    def estimate_bpm(self):
        if len(self.rgb_signals) < self.buffer_size:
            return None # Non abbiamo accumulato abbastanza dati

        # Calcolo dell'effettivo FPS in base al tempo reale
        time_diff = self.times[-1] - self.times[0]
        if time_diff == 0: return None
        actual_fps = len(self.times) / time_diff

        signals = np.array(self.rgb_signals) # shape: (N, 3)
        
        # Normalizzazione del segnale
        signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)

        # Applica FastICA (Implementazione Python equivalente al codice MATLAB del prof)
        try:
            source_signals = self.ica.fit_transform(signals)
        except:
            return None

        # Filtro passa-banda Butterworth per i battiti divini (0.75 Hz = 45 BPM, 3.0 Hz = 180 BPM)
        nyquist = 0.5 * actual_fps
        low = 0.75 / nyquist
        high = 3.0 / nyquist
        if low <= 0 or high >= 1:
            return None
            
        b, a = butter(3, [low, high], btype='band')
        
        max_power = 0
        best_bpm = 0
        
        # Processa ciascuna delle 3 componenti indipendenti estratte dall'ICA
        for i in range(source_signals.shape[1]):
            comp = source_signals[:, i]
            # Filtra la componente
            filtered = filtfilt(b, a, comp)
            
            # Trasformata di Fourier per trovare la frequenza dominante
            fft_data = np.fft.rfft(filtered)
            fft_freq = np.fft.rfftfreq(len(filtered), 1.0 / actual_fps)
            
            power = np.abs(fft_data)
            
            # Cerca il picco massimo nel range 0.75-3.0 Hz (45-180 BPM)
            valid_idx = np.where((fft_freq >= 0.75) & (fft_freq <= 3.0))
            if len(valid_idx[0]) > 0:
                peak_idx = valid_idx[0][np.argmax(power[valid_idx])]
                peak_freq = fft_freq[peak_idx]
                peak_power = power[peak_idx]
                
                if peak_power > max_power:
                    max_power = peak_power
                    best_bpm = peak_freq * 60.0 # hz to bpm
                    
        return int(best_bpm) if best_bpm > 0 else None

def main():
    # Controlla e scarica il file .task se mancante, in modo da poter consegnare solo questo .py
    download_model_if_needed()

    print("Inizializzazione DMS (Driver Monitoring System)...")
    print("Premi 'q' per uscire.")
    
    # Inizializza MediaPipe FaceLandmarker con le nuove API Data/Tasks (Compatibile con Python 3.13)
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Usa il modello scaricato localmente
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO)
    
    # Inizializzazione della videocamera con landmarker
    landmarker = FaceLandmarker.create_from_options(options)

    # Inizializza la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Errore: impossibile accedere alla telecamera.")
        return

    # Tracking Gufo (Distrazione Posizione Testa)
    is_looking_away = False
    away_start_time = None
    focus_start_time_owl = None
    long_owl_active = False
    short_owl_active = False
    history = deque()
    
    # Tracking Occhi (Sleep / Microsleep)
    eyes_closed = False
    eyes_closed_start_time = None
    eyes_open_start_time = None
    microsleep_active = False
    sleep_active = False

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()
        delta_t = current_time - last_time
        last_time = current_time
        img_h, img_w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Converte l'immagine per le nuove API di mediapipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(current_time * 1000)
        face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if face_landmarker_result.face_landmarks:
            for face_landmarks in face_landmarker_result.face_landmarks:
                # 1. Controllo Distrazione Gufo (Usa la variabile globale HEAD_POSE_THRESHOLD)
                pitch, yaw, roll = get_head_pose(face_landmarks, img_w, img_h)
                if abs(yaw) > HEAD_POSE_THRESHOLD or abs(pitch) > HEAD_POSE_THRESHOLD:
                    is_looking_away = True
                else:
                    is_looking_away = False
                    
                # 2. Controllo Occhi Chiusi (EAR) -- Usa la variabile globale EAR_THRESHOLD
                ear = get_ear(face_landmarks, img_w, img_h)
                if ear < EAR_THRESHOLD:
                    eyes_closed = True
                else:
                    eyes_closed = False
        else:
            is_looking_away = True
            eyes_closed = False  # Se non c'è volto non allarmiamo sleep, ma assenza.

        # --- TICKER OCCHI (Microsleep / Sleep) ---
        if eyes_closed:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = current_time
            eyes_open_start_time = None
        else:
            if eyes_open_start_time is None:
                eyes_open_start_time = current_time
            eyes_closed_start_time = None

        # v. Microsonno / vi. Sonno (Usando le variabili)
        if eyes_closed_start_time is not None:
            closed_duration = current_time - eyes_closed_start_time
            if closed_duration >= TIMER_SLEEP:
                sleep_active = True
                microsleep_active = False # Promosso a sleep
            elif closed_duration >= TIMER_MICROSLEEP and not sleep_active:
                microsleep_active = True

        # Disattivazione Sonno/Microsonno (Usando la variabile)
        if (microsleep_active or sleep_active) and not eyes_closed and eyes_open_start_time is not None:
            if (current_time - eyes_open_start_time) >= TIMER_RESET_EYES:
                microsleep_active = False
                sleep_active = False

        # --- TICKER GUFO (Long / Short Distraction) ---
        if is_looking_away:
            if away_start_time is None:
                away_start_time = current_time
            focus_start_time_owl = None
        else:
            if focus_start_time_owl is None:
                focus_start_time_owl = current_time
            away_start_time = None

        history.append((current_time, delta_t, is_looking_away))
        
        # Usa TIMER_SHORT_OWL_WINDOW
        while history and history[0][0] < current_time - TIMER_SHORT_OWL_WINDOW:
            history.popleft()
            
        cumulative_away_time = sum(h[1] for h in history if h[2])

        # Long Owl
        if away_start_time is not None and (current_time - away_start_time) >= TIMER_LONG_OWL:
            long_owl_active = True
        if not is_looking_away:
            long_owl_active = False

        # Short Owl
        if cumulative_away_time >= TIMER_SHORT_OWL_CUMULATIVE:
            short_owl_active = True
            
        # Disattivazione Distrazione
        if short_owl_active and not is_looking_away and focus_start_time_owl is not None:
            if (current_time - focus_start_time_owl) >= TIMER_RESET_OWL:
                short_owl_active = False
                history.clear()

        # --- PRIORITA' OUTPUT VISUALE ---
        # e. Possibili stati del conducente, secondo la priorità d'emergenza
        status_text = "Focused on the road"
        status_color = (0, 255, 0)

        if sleep_active:
            status_text = "Sleep"
            status_color = (0, 0, 255) # Rosso estremo
        elif microsleep_active:
            status_text = "Microsleep"
            status_color = (255, 0, 255) # Magenta
        elif long_owl_active:
            status_text = "Distracted (long)"
            status_color = (0, 165, 255) # Arancione scuro
        elif short_owl_active:
            status_text = "Distracted (short)"
            status_color = (0, 255, 255) # Giallo

        # d., e. L'output deve essere un video del volto con angoli in basso
        cv2.putText(frame, f"Status: {status_text}", (img_w - 350, img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
        # f. Testo in basso a sinistra con il battito cardiaco in BPM
        cv2.putText(frame, "Heart Rate: -- BPM", (20, img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("DMS - Driver Monitoring System", frame)

        # Interrompi con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()