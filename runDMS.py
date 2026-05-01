import cv2
import numpy as np

def main():
    print("Inizializzazione DMS (Driver Monitoring System)...")
    print("Premi 'q' per uscire.")
    
    # Inizializza la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Errore: impossibile accedere alla telecamera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # TODO: Implementare MediaPipe / OpenCV per:
        # a. Rilevare occhi e naso
        # b. Identificare le distrazioni (Owl long/short, Lizard long/short) e stati degli occhi
        # c. Calcolare il battito cardiaco con fastICA
        
        # d. L'output deve essere un video del volto
        # e. Testo in basso a destra con lo stato del conducente
        cv2.putText(frame, "Status: Focused on the road", (frame.shape[1] - 300, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        # f. Testo in basso a sinistra con il battito cardiaco in BPM
        cv2.putText(frame, "Heart Rate: -- BPM", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("DMS - Driver Monitoring System", frame)

        # Interrompi con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()