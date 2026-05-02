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

# Soglie TESTA (Gufo - Distrazione)
YAW_BASELINE = 18.5            # Il tuo "Zero" per la rotazione dx/sx
PITCH_BASELINE = 175.0         # Il tuo "Zero" per la rotazione su/giù
YAW_THRESHOLD = 15.0           # Gradi di tolleranza dal punto zero (Dx/Sx)
PITCH_THRESHOLD = 12.0         # Gradi di tolleranza dal punto zero (Su/Giù)

# Soglie SGUARDO (Lucertola - Distrazione)
# I valori ruotano attorno allo 0.5 (centro esatto).
GAZE_H_MIN = 0.40              # Troppo a sinistra
GAZE_H_MAX = 0.60              # Troppo a destra
GAZE_V_MIN = 0.35              # Troppo in alto
GAZE_V_MAX = 0.65              # Troppo in basso (tipico del cellulare)

# Soglie OCCHI
EAR_THRESHOLD = 0.18           
TIMER_MICROSLEEP = 4.0         
TIMER_SLEEP = 7.0              
TIMER_RESET_EYES = 2.0         # Prof req: mantenuti aperti per almeno 2s

# Soglie TEMPORALI (Gufo e Lucertola condividono le stesse regole temporali)
TIMER_LONG_DISTR = 5.0               # Prof req: 5s per distrazione lunga
TIMER_SHORT_DISTR_CUMULATIVE = 10.0  # Prof req: 10s cumulativi
TIMER_SHORT_DISTR_WINDOW = 30.0      # Prof req: in una finestra di 30s
TIMER_RESET_DISTR = 0.5              # 0.5s per spezzare la distrazione continua
TIMER_VISUAL_HOLD = 2.0              # TEMPO DI VISUALIZZAZIONE a schermo prima di far sparire l'allarme

# Tolleranza BPM
MAX_BPM_VARIATION = 20.0       # Variazione max per filtro anomalie
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

def get_gaze_ratio(face_landmarks, img_w, img_h):
    # Ritorna la posizione dell'iride (Orizzontale, Verticale). Valore ~0.5 = centro.
    def dist(p1_idx, p2_idx):
        x1, y1 = face_landmarks[p1_idx].x * img_w, face_landmarks[p1_idx].y * img_h
        x2, y2 = face_landmarks[p2_idx].x * img_w, face_landmarks[p2_idx].y * img_h
        return np.hypot(x1 - x2, y1 - y2)

    # Occhio destro (per la telecamera è a sinistra). Esterno: 33, Interno: 133, Iride: 468
    width_r = dist(33, 133)
    iris_h_r = dist(468, 33) / width_r if width_r > 0 else 0.5
    height_r = dist(159, 145)
    iris_v_r = dist(468, 159) / height_r if height_r > 0 else 0.5

    # Occhio sinistro (per la telecamera è a destra). Interno: 362, Esterno: 263, Iride: 473
    width_l = dist(362, 263)
    iris_h_l = dist(473, 362) / width_l if width_l > 0 else 0.5
    height_l = dist(386, 374)
    iris_v_l = dist(473, 386) / height_l if height_l > 0 else 0.5

    # Media tra i due occhi
    gaze_h = (iris_h_r + iris_h_l) / 2.0
    gaze_v = (iris_v_r + iris_v_l) / 2.0
    
    return gaze_h, gaze_v

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

    # Variabili Gufo (Testa)
    is_owl = False
    owl_start_time = None
    owl_focus_start_time = None
    long_owl_active = False
    short_owl_active = False
    owl_history = deque()
    cumulative_owl_time = 0.0

    # Variabili Lucertola (Sguardo)
    is_lizard = False
    lizard_start_time = None
    lizard_focus_start_time = None
    long_lizard_active = False
    short_lizard_active = False
    lizard_history = deque()
    cumulative_lizard_time = 0.0
    
    # Variabili Sonno
    eyes_closed = False
    eyes_closed_start_time = None
    eyes_open_start_time = None
    microsleep_active = False
    sleep_active = False

    last_time = time.time()
    ear = yaw = pitch = gaze_h = gaze_v = 0.0 
    
    # --- STIMATORE BATTITO CARDIACO ---
    hr_estimator = HeartRateEstimator(buffer_size=150)
    current_bpm = None
    last_bpm_calc_time = time.time()
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
                    pitch, yaw, roll = get_head_pose(face_landmarks, img_w, img_h)
                    ear = get_ear(face_landmarks, img_w, img_h)
                    
                    # Estrazione sguardo (solo se abbiamo i landmark dell'iride)
                    if len(face_landmarks) > 473:
                        gaze_h, gaze_v = get_gaze_ratio(face_landmarks, img_w, img_h)
                    
                    # Controllo Gufo (Movimento Testa)
                    diff_yaw = min(abs(yaw - YAW_BASELINE), 360 - abs(yaw - YAW_BASELINE))
                    diff_pitch = min(abs(pitch - PITCH_BASELINE), 360 - abs(pitch - PITCH_BASELINE))
                    
                    is_head_distracted = (diff_yaw > YAW_THRESHOLD or diff_pitch > PITCH_THRESHOLD)
                    is_owl = is_head_distracted

                    # Controllo Lucertola (Testa Dritta MA Sguardo deviato)
                    is_gaze_distracted = (gaze_h < GAZE_H_MIN or gaze_h > GAZE_H_MAX or gaze_v < GAZE_V_MIN or gaze_v > GAZE_V_MAX)
                    
                    if not is_head_distracted and is_gaze_distracted:
                        is_lizard = True
                    else:
                        is_lizard = False
                        
                    # Controllo Sonno
                    if ear < EAR_THRESHOLD:
                        eyes_closed = True
                    else:
                        eyes_closed = False
                        
                    hr_estimator.add_frame(frame, face_landmarks, current_time, img_w, img_h)
            else:
                is_owl = True
                is_lizard = False
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
                if (current_time - eyes_open_start_time) >= TIMER_VISUAL_HOLD:
                    microsleep_active = False
                    sleep_active = False

            # --- TICKER GUFO ---
            if is_owl:
                if owl_start_time is None:
                    owl_start_time = current_time
                owl_focus_start_time = None
            else:
                if owl_focus_start_time is None:
                    owl_focus_start_time = current_time
                
                # Resetta il contatore interno della distrazione dopo 0.5s
                if (current_time - owl_focus_start_time) >= TIMER_RESET_DISTR:
                    owl_start_time = None
                
                # Mantiene l'allarme visivo su schermo per TIMER_VISUAL_HOLD (2.0s)
                if long_owl_active and (current_time - owl_focus_start_time) >= TIMER_VISUAL_HOLD:
                    long_owl_active = False

            owl_history.append((current_time, delta_t, is_owl))
            while owl_history and owl_history[0][0] < current_time - TIMER_SHORT_DISTR_WINDOW:
                owl_history.popleft()
            cumulative_owl_time = sum(h[1] for h in owl_history if h[2])

            if owl_start_time is not None and (current_time - owl_start_time) >= TIMER_LONG_DISTR:
                long_owl_active = True
            if cumulative_owl_time >= TIMER_SHORT_DISTR_CUMULATIVE:
                short_owl_active = True
                
            if short_owl_active and not is_owl and owl_focus_start_time is not None:
                if (current_time - owl_focus_start_time) >= TIMER_VISUAL_HOLD:
                    short_owl_active = False
                    owl_history.clear()

            # --- TICKER LUCERTOLA ---
            if is_lizard:
                if lizard_start_time is None:
                    lizard_start_time = current_time
                lizard_focus_start_time = None
            else:
                if lizard_focus_start_time is None:
                    lizard_focus_start_time = current_time
                
                # Resetta il contatore interno della distrazione dopo 0.5s
                if (current_time - lizard_focus_start_time) >= TIMER_RESET_DISTR:
                    lizard_start_time = None
                
                # Mantiene l'allarme visivo su schermo per TIMER_VISUAL_HOLD (2.0s)
                if long_lizard_active and (current_time - lizard_focus_start_time) >= TIMER_VISUAL_HOLD:
                    long_lizard_active = False

            lizard_history.append((current_time, delta_t, is_lizard))
            while lizard_history and lizard_history[0][0] < current_time - TIMER_SHORT_DISTR_WINDOW:
                lizard_history.popleft()
            cumulative_lizard_time = sum(h[1] for h in lizard_history if h[2])

            if lizard_start_time is not None and (current_time - lizard_start_time) >= TIMER_LONG_DISTR:
                long_lizard_active = True
            if cumulative_lizard_time >= TIMER_SHORT_DISTR_CUMULATIVE:
                short_lizard_active = True
                
            if short_lizard_active and not is_lizard and lizard_focus_start_time is not None:
                if (current_time - lizard_focus_start_time) >= TIMER_VISUAL_HOLD:
                    short_lizard_active = False
                    lizard_history.clear()

            # --- CALCOLO BPM ---
            if current_time - last_bpm_calc_time > 1.0:
                bpm_estimate = hr_estimator.estimate_bpm()
                if bpm_estimate is not None:
                    if 45 <= bpm_estimate <= 180:
                        if len(bpm_history) == 0:
                            bpm_history.append(bpm_estimate)
                            current_bpm = bpm_estimate
                        else:
                            current_mean = np.mean(bpm_history)
                            if abs(bpm_estimate - current_mean) <= MAX_BPM_VARIATION:
                                bpm_history.append(bpm_estimate)
                                current_bpm = int(np.mean(bpm_history))
                            else:
                                print(f"-> Anomalia BPM ignorata: {bpm_estimate} (Media={current_mean:.1f})")
                last_bpm_calc_time = current_time

            # --- DEBUG CONSOLE MIGLIORATO ---
            cont_owl_time = (current_time - owl_start_time) if owl_start_time is not None else 0.0
            cont_lizard_time = (current_time - lizard_start_time) if lizard_start_time is not None else 0.0
            
            # Pulisce un po' visivamente separando i frame
            print(f"\n{'='*55}")
            
            # --- Output Gufo ---
            stato_gufo = "⚠️ DISTRATTO " if is_owl else "✅ ATTENTO   "
            print(f"[ TESTA (Gufo) ]    Stato: {stato_gufo} | Yaw: {yaw:>5.1f}°")
            print(f"  -> Timer Continuo:   {cont_owl_time:>4.1f}s / {TIMER_LONG_DISTR}s")
            print(f"  -> Timer Cumulativo: {cumulative_owl_time:>4.1f}s / {TIMER_SHORT_DISTR_CUMULATIVE}s (negli ultimi 30s)")

            # --- Output Lucertola ---
            stato_lucertola = "⚠️ DISTRATTO " if is_lizard else "✅ ATTENTO   "
            print(f"\n[ SGUARDO (Lucertola)] Stato: {stato_lucertola} | Gaze: (H:{gaze_h:.2f}, V:{gaze_v:.2f})")
            print(f"  -> Timer Continuo:   {cont_lizard_time:>4.1f}s / {TIMER_LONG_DISTR}s")
            print(f"  -> Timer Cumulativo: {cumulative_lizard_time:>4.1f}s / {TIMER_SHORT_DISTR_CUMULATIVE}s (negli ultimi 30s)")
            print(f"{'='*55}")

            # --- OUTPUT VISUALE ---
            # 1. Output Standard (Sonno e Gufo)
            if sleep_active:
                status_text = "Sleep"
                status_color = (0, 0, 255)
            elif microsleep_active:
                status_text = "Microsleep"
                status_color = (255, 0, 255)
            elif long_owl_active:
                status_text = "Owl Distracted (long)"
                status_color = (0, 165, 255)
            elif short_owl_active:
                status_text = "Owl Distracted (short)"
                status_color = (0, 255, 255)
            else:
                status_text = "Focused on the road"
                status_color = (0, 255, 0)

            cv2.putText(frame, status_text, (img_w - 300, img_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                        
            # 2. Output Debug Lucertola (Sopra l'output standard)
            lizard_text = ""
            lizard_color = (255, 255, 255) # Bianco default
            
            if long_lizard_active:
                lizard_text = "LIZARD: Long Distraction!"
                lizard_color = (200, 0, 200) # Viola/Rosa BGR
            elif short_lizard_active:
                lizard_text = "LIZARD: Short Distraction!"
                lizard_color = (255, 100, 100) # Azzurro BGR
            elif is_lizard:
                lizard_text = f"LIZARD: Looking away... ({gaze_h:.2f}, {gaze_v:.2f})"
                lizard_color = (200, 200, 200) # Grigio
                
            if lizard_text:
                cv2.putText(frame, lizard_text, (img_w - 350, img_h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, lizard_color, 2)

            # BPM
            if current_bpm is not None:
                bpm_text = f"Heart Rate: {current_bpm} BPM"
            else:
                bpm_text = "Heart Rate: -- BPM (Calcolo...)"
                
            cv2.putText(frame, bpm_text, (20, img_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("DMS - Driver Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n\nInterruzione forzata rilevata (CTRL+C).")
    finally:
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Chiusura completata con successo. A presto!")

if __name__ == "__main__":
    main()