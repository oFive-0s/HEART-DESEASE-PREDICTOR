import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pyttsx3
import platform
import os
import time
from twilio.rest import Client

# Twilio configuration (replace with your actual credentials)
account_sid = "ACff002a2c4cbece635eee3d1fb13fddf0"
auth_token = "9c00e6ae29fc4d50a80670d0121be8d2"
twilio_client = Client(account_sid, auth_token)
twilio_phone_number = "6361053308"
emergency_number = "9480443130"  # Replace with the actual emergency contact number

# Function to send SMS alerts via Twilio
def send_sms_alert(message):
    try:
        twilio_client.messages.create(
            body=message,
            from_= 8792106534,
            to= 8296092263
        )
        print("SMS alert sent successfully.")
    except Exception as e:
        print(f"Failed to send SMS alert: {e}")

# Function to make a phone call via Twilio
def make_phone_call():
    try:
        call = twilio_client.calls.create(
            twiml='<Response><Say>Anomaly detected. Please take immediate action.</Say></Response>',
            from_="8792106534",
            to="8296092263"
        )
        print(f"Phone call initiated: SID {call.sid}")
    except Exception as e:
        print(f"Failed to initiate phone call: {e}")

# Function to play a beep sound
def beep_alert():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms
    else:
        os.system("echo -e '\a'")  # Beep sound for macOS/Linux

# Function to speak the alert message
def voice_alert(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

# Simulate synthetic ECG and heart rate data (replace with real-time wearable data)
np.random.seed(int(time.time()))
data_size = 1500
ecg_data = np.sin(np.linspace(0, 30 * np.pi, data_size)) + 0.1 * np.random.randn(data_size)  # Synthetic ECG
heart_rate_data = np.random.normal(loc=75, scale=8, size=data_size)  # Synthetic heart rate data

# Preprocess ECG signal by filtering out high-frequency noise (Butterworth filter)
def preprocess_ecg(ecg_signal):
    b, a = signal.butter(3, 0.05)  # Low-pass filter to remove high-frequency noise
    ecg_filtered = signal.filtfilt(b, a, ecg_signal)
    return ecg_filtered

ecg_filtered = preprocess_ecg(ecg_data)

# Compute HRV features: SDNN (standard deviation of RR intervals) and RMSSD
def compute_hrv(ecg_signal, sampling_rate=1):
    distance = max(int(sampling_rate * 0.6), 1)
    peaks, _ = signal.find_peaks(ecg_signal, distance=distance)  # Find R-peaks in ECG
    rr_intervals = np.diff(peaks) * sampling_rate  # R-R intervals in seconds
    hrv_sdnn = np.std(rr_intervals)  # Standard deviation of RR intervals (SDNN)
    hrv_rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))  # RMSSD
    return hrv_sdnn, hrv_rmssd, rr_intervals

hrv_sdnn, hrv_rmssd, rr_intervals = compute_hrv(ecg_filtered)

# Feature extraction
mean_ecg = np.mean(ecg_filtered)
std_ecg = np.std(ecg_filtered)
max_ecg = np.max(ecg_filtered)
min_ecg = np.min(ecg_filtered)

features = np.column_stack([
    heart_rate_data,
    np.full(data_size, mean_ecg),
    np.full(data_size, std_ecg),
    np.full(data_size, max_ecg),
    np.full(data_size, min_ecg)
])

# Generate synthetic labels
labels = np.array([0 if i % 50 != 0 else 1 for i in range(data_size)])  # 1 every 50th sample for anomaly

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an anomaly detection model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train_scaled)

# Predict anomalies
y_pred = model.predict(X_test_scaled)
y_pred = np.where(y_pred == 1, 0, 1)

# Check for anomalies and send alerts
anomaly_indices = np.where(y_pred == 1)[0]
if len(anomaly_indices) > 0:
    alert_message = "Anomaly detected! Potential silent heart attack."
    beep_alert()
    voice_alert(alert_message)
    send_sms_alert(alert_message)
    make_phone_call()  # Initiate a phone call

# Evaluate the model
print("Anomaly Detection Results:")
print(classification_report(y_test, y_pred))

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ecg_filtered, label='Filtered ECG Signal')
plt.title('Filtered ECG with Detected Anomalies')
plt.scatter(anomaly_indices, ecg_filtered[anomaly_indices], color='red', label='Anomalies')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(heart_rate_data, label='Heart Rate')
plt.scatter(anomaly_indices, heart_rate_data[anomaly_indices], color='red', label='Anomalies')
plt.title('Heart Rate with Detected Anomalies')
plt.legend()

plt.tight_layout()
plt.show()