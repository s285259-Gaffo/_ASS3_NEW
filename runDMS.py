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

# Soglie TESTA (Distrazione)
YAW_BASELINE = 18.5            # Il tuo "Zero" per la rotazione dx/sx
PITCH_BASELINE = 175.0         # Il tuo "Zero" per la rotazione su/giù

YAW_THRESHOLD = 15.0           # Gradi di tolleranza dal punto zero (Dx/Sx)
PITCH_THRESHOLD = 12.0         # Gradi di tolleranza dal punto zero (Su/Giù)

# Soglie OCCHI
EAR_THRESHOLD = 0.18           
TIMER_MICROSLEEP = 4.0         
TIMER_SLEEP = 7.0              
TIMER_RESET_EYES = 2.0         # Prof req: mantenuti aperti per almeno 2s (regole v e vi)

# Soglie TESTA (Distrazione)
TIMER_LONG_OWL = 5.0               
TIMER_SHORT_OWL_CUMULATIVE = 10.0  # Prof req: 10s
TIMER_SHORT_OWL_WINDOW = 30.0      # Prof req: in una finestra di 30s
TIMER_RESET_OWL = 0.5          # Tempo di sguardo dritto necessario per spezzare la distrazione continua

# Tolleranza BPM
MAX_BPM_VARIATION = 20.0       # Variazione massima accettata rispetto alla media attuale (in BPM)
# -------------------------------------

def download_model_if_needed():
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"Modello MediaPipe '{model_path}' non trovato.")
        print("Download in corso (circa 3MB)... attendere prego.")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download completato con successo!")

def get_head_pose(face_landmarks, img_w, img_h):
    face3d = np.array([
        (0.0, 0.0, 0.0),            
        (0.0, -330.0, -65.0),       
        (-225.0, 170.0, -135.0),    
        (225.0, 170.0, -135.0),     
        (-150.0, -150.0, -125.0),   
        (150.0, -150.0, -125.0)     
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
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))
    pitch, yaw, roll = eulerAngles[0][0], eulerAngles[1][0], eulerAngles[2][0]

    return pitch, yaw, roll

def get_ear(face_landmarks, img_w, img_h):
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
        x = int(face_landmarks[10].x * img_w)
        y = int(face_landmarks[10].y * img_h)
        box_size = 10
        y1, y2 = max(0, y - box_size), min(img_h, y + box_size)
        x1, x2 = max(0, x - box_size), min(img_w, x + box_size)
        
        if y2 > y1 and x2 > x1:
            roi = image[y1:y2, x1:x2]
            avg_color = np.mean(roi, axis=(0, 1)) 
            self.rgb_signals.append(avg_color)
            self.times.append(current_time)
            
        if len(self.rgb_signals) > self.buffer_size:
            self.rgb_signals.pop(0)
            self.times.pop(0)

    def estimate_bpm(self):
        if len(self.rgb_signals) < self.buffer_size:
            return None 

        time_diff = self.times[-1] - self.times[0]
        if time_diff == 0: return None
        actual_fps = len(self.times) / time_diff

        signals = np.array(self.rgb_signals) 
        signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)

        try:
            source_signals = self.ica.fit_transform(signals)
        except:
            return None

        nyquist = 0.5 * actual_fps
        low = 0.75 / nyquist
        high = 3.0 / nyquist
        if low <= 0 or high >= 1:
            return None
            
        b, a = butter(3, [low, high], btype='band')
        
        max_power = 0
        best_bpm = 0
        
        for i in range(source_signals.shape[1]):
            comp = source_signals[:, i]
            filtered = filtfilt(b, a, comp)
            
            fft_data = np.fft.rfft(filtered)
            fft_freq = np.fft.rfftfreq(len(filtered), 1.0 / actual_fps)
            power = np.abs(fft_data)
            
            valid_idx = np.where((fft_freq >= 0.75) & (fft_freq <= 3.0))
            if len(valid_idx[0]) > 0:
                peak_idx = valid_idx[0][np.argmax(power[valid_idx])]
                peak_freq = fft_freq[peak_idx]
                peak_power = power[peak_idx]
                
                if peak_power > max_power:
                    max_power = peak_power
                    best_bpm = peak_freq * 60.0
                    
        return int(best_bpm) if best_bpm > 0 else None

def main():
    download_model_if_needed()

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO)
    
    landmarker = FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Errore: impossibile accedere alla telecamera.")
        return

    is_looking_away = False
    away_start_time = None
    focus_start_time_owl = None
    long_owl_active = False
    short_owl_active = False
    history = deque()
    
    eyes_closed = False
    eyes_closed_start_time = None
    eyes_open_start_time = None
    microsleep_active = False
    sleep_active = False

    last_time = time.time()
    ear = yaw = pitch = 0.0 
    cumulative_away_time = 0.0
    
    # --- STIMATORE BATTITO CARDIACO ---
    hr_estimator = HeartRateEstimator(buffer_size=150)
    current_bpm = None
    last_bpm_calc_time = time.time()
    
    # Memoria circolare per gli ultimi 5 valori VALIDI di BPM
    bpm_history = deque(maxlen=5)
    
    print("\nSistema di monitoraggio avviato in modo sicuro.")
    print(" -> Premi 'q' sulla finestra del video per uscire.")
    print(" -> Premi CTRL+C sul terminale per interrompere.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            current_time = time.time()
            delta_t = current_time - last_time
            last_time = current_time
            img_h, img_w, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(current_time * 1000)
            face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if face_landmarker_result.face_landmarks:
                for face_landmarks in face_landmarker_result.face_landmarks:
                    # Rilevamento dati Greci (Angoli e EAR)
                    pitch, yaw, roll = get_head_pose(face_landmarks, img_w, img_h)
                    ear = get_ear(face_landmarks, img_w, img_h)
                    
                    # Calcoliamo di quanti gradi ti sei mosso rispetto al TUO punto zero
                    diff_yaw = min(abs(yaw - YAW_BASELINE), 360 - abs(yaw - YAW_BASELINE))
                    diff_pitch = min(abs(pitch - PITCH_BASELINE), 360 - abs(pitch - PITCH_BASELINE))

                    # Se la differenza supera la tolleranza, sei distratto
                    if diff_yaw > YAW_THRESHOLD or diff_pitch > PITCH_THRESHOLD:
                        is_looking_away = True
                    else:
                        is_looking_away = False
                        
                    # Controllo Sonno
                    if ear < EAR_THRESHOLD:
                        eyes_closed = True
                    else:
                        eyes_closed = False
                        
                    # Inseriamo il frame corrente e i landmarks nello stimatore
                    hr_estimator.add_frame(frame, face_landmarks, current_time, img_w, img_h)
            else:
                is_looking_away = True
                eyes_closed = False  

            # --- TICKER OCCHI (Microsleep / Sleep) ---
            if eyes_closed:
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = current_time
                eyes_open_start_time = None
            else:
                if eyes_open_start_time is None:
                    eyes_open_start_time = current_time
                eyes_closed_start_time = None

            if eyes_closed_start_time is not None:
                closed_duration = current_time - eyes_closed_start_time
                if closed_duration >= TIMER_SLEEP:
                    sleep_active = True
                    microsleep_active = False 
                elif closed_duration >= TIMER_MICROSLEEP and not sleep_active:
                    microsleep_active = True

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
                
                # Resetta il contatore continuità solo se hai guardato dritto per TIMER_RESET_OWL
                if (current_time - focus_start_time_owl) >= TIMER_RESET_OWL:
                    away_start_time = None
                    long_owl_active = False

            # Gestione Storico (Finestra cumulativa 30s)
            history.append((current_time, delta_t, is_looking_away))
            while history and history[0][0] < current_time - TIMER_SHORT_OWL_WINDOW:
                history.popleft()
                
            cumulative_away_time = sum(h[1] for h in history if h[2])

            # Attivazione Long Owl
            if away_start_time is not None and (current_time - away_start_time) >= TIMER_LONG_OWL:
                long_owl_active = True

            # Attivazione Short Owl (Cumulativa)
            if cumulative_away_time >= TIMER_SHORT_OWL_CUMULATIVE:
                short_owl_active = True
                
            # Disattivazione visiva dell'allarme Short Owl (serve guardare la strada per 2 secondi per spegnerlo e pulire la cronologia)
            if short_owl_active and not is_looking_away and focus_start_time_owl is not None:
                if (current_time - focus_start_time_owl) >= 2.0:
                    short_owl_active = False
                    history.clear()
                    
            # --- CALCOLO BPM CON MEDIA MOBILE E RIFIUTO ANOMALIE ---
            if current_time - last_bpm_calc_time > 1.0:
                bpm_estimate = hr_estimator.estimate_bpm()
                
                if bpm_estimate is not None:
                    # Validazione fisiologica di base
                    if 45 <= bpm_estimate <= 180:
                        
                        if len(bpm_history) == 0:
                            # Primo dato utile: lo accettiamo direttamente
                            bpm_history.append(bpm_estimate)
                            current_bpm = bpm_estimate
                        else:
                            # Calcoliamo la media attuale
                            current_mean = np.mean(bpm_history)
                            
                            # Rifiutiamo anomalie: il nuovo dato deve essere entro la tolleranza
                            if abs(bpm_estimate - current_mean) <= MAX_BPM_VARIATION:
                                bpm_history.append(bpm_estimate)
                                # Aggiorniamo il BPM da mostrare a schermo con la nuova media
                                current_bpm = int(np.mean(bpm_history))
                            else:
                                print(f"-> Anomalia BPM ignorata: Stima={bpm_estimate}, Media attuale={current_mean:.1f}")
                                
                last_bpm_calc_time = current_time

            # --- DEBUG CONSOLE (Timer inclusi) ---
            cont_distr_time = (current_time - away_start_time) if away_start_time is not None else 0.0
            
            occhi_str = "CHIUSI" if eyes_closed else "APERTI"
            distr_str = "SI" if is_looking_away else "NO"
            
            print(f"EAR: {ear:.2f} | YAW: {yaw:>5.1f}° | Occhi: {occhi_str:<6} | Distr: {distr_str:<2} | Continuo (Long): {cont_distr_time:.1f}s/5s | Cumulativo (Short): {cumulative_away_time:.1f}s/10s")

            # --- OUTPUT VISUALE UNIFICATO CON PRIORITÀ ---
            if sleep_active:
                status_text = "Sleep"
                status_color = (0, 0, 255) # Rosso BGR
            elif microsleep_active:
                status_text = "Microsleep"
                status_color = (255, 0, 255) # Magenta BGR
            elif long_owl_active:
                status_text = "Distracted (long)"
                status_color = (0, 165, 255) # Arancione BGR
            elif short_owl_active:
                status_text = "Distracted (short)"
                status_color = (0, 255, 255) # Giallo BGR
            else:
                status_text = "Focused on the road"
                status_color = (0, 255, 0) # Verde BGR

            cv2.putText(frame, status_text, (img_w - 300, img_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        
            # Stampa i BPM in basso a sinistra
            if current_bpm is not None:
                bpm_text = f"Heart Rate: {current_bpm} BPM"
            else:
                bpm_text = "Heart Rate: -- BPM (Calcolo...)"
                
            cv2.putText(frame, bpm_text, (20, img_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("DMS - Driver Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nUscita manuale richiesta (tasto 'q').")
                break

    except KeyboardInterrupt:
        print("\n\nInterruzione forzata rilevata (CTRL+C).")
        
    finally:
        print("Spegnimento della videocamera e pulizia delle finestre...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Chiusura completata con successo. A presto!")

if __name__ == "__main__":
    main()