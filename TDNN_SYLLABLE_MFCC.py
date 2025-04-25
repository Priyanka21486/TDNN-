import numpy as np
import librosa
import os
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
import numpy as np
import librosa
import gammatone.filters
from scipy.signal import hilbert
import matplotlib.pylab as pl
import tensorflow as tf
from tqdm import tqdm

def extract_mfcc_from_syllables(audio_path, syllable_boundaries, sr=16000, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc_features = []
    
    for start, end in syllable_boundaries:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        syllable_audio = y[start_sample:end_sample]
        mfcc = librosa.feature.mfcc(y=syllable_audio, sr=sr, n_fft=1024,  n_mfcc=n_mfcc)
        mean=np.mean(mfcc, axis=1)
        #print(f"initial sylabbic mfcc is :::::::::::::::{mean.shape}")
        mfcc_features.append(mean.T)  # Transpose for correct shape (time_steps, n_mfcc)
    
    return mfcc_features
def syllable_boundaries_function(file_path):
    def peakdet(v, delta, x=None):
        maxtab = []
        mintab = []

        if x is None:
            x = np.arange(len(v))

        v = np.asarray(v)
        if len(v) != len(x):
            raise ValueError("Input vectors v and x must have the same length")

        if not np.isscalar(delta):
            raise ValueError("Input argument delta must be a scalar")

        if delta <= 0:
            raise ValueError("Input argument delta must be positive")

        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN

        lookformax = True
        for i in np.arange(len(v)):
            this = v[i]
            if this > mx or np.isnan(mx):
                mx = this
                mxpos = x[i]
            if this < mn or np.isnan(mn):
                mn = this
                mnpos = x[i]

            if lookformax:
                if this < mx - delta:
                    # Convert to scalar before appending
                    maxtab.append((mxpos, mx.item()))
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn + delta:
                    # Convert to scalar before appending
                    mintab.append((mnpos, mn.item()))
                    mx = this
                    mxpos = x[i]
                    lookformax = True

        # Convert lists to numpy arrays
        maxtab = np.array(maxtab)
        mintab = np.array(mintab)

        return maxtab, mintab

# Define the theta oscillator function
    def thetaOscillator(ENVELOPE, f=5, Q=0.5, thr=0.025, verbose=1):
        N = 10  # How many most energetic bands to use (default = 8)

        if N > ENVELOPE.size:
            print('WARNING: Input dimensionality smaller than the N parameter. Using all frequency bands.')

        a = np.array([
            [72, 34, 22, 16, 12, 9, 8, 6, 5, 4, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0],
            [107, 52, 34, 25, 19, 16, 13, 11, 10, 9, 8, 7, 6, 5, 5, 4, 4, 4, 3, 3],
            [129, 64, 42, 31, 24, 20, 17, 14, 13, 11, 10, 9, 8, 7, 7, 6, 6, 5, 5, 4],
            [145, 72, 47, 35, 28, 23, 19, 17, 15, 13, 12, 10, 9, 9, 8, 7, 7, 6, 6, 5],
            [157, 78, 51, 38, 30, 25, 21, 18, 16, 14, 13, 12, 11, 10, 9, 8, 8, 7, 7, 6],
            [167, 83, 55, 41, 32, 27, 23, 19, 17, 15, 14, 12, 11, 10, 10, 9, 8, 8, 7, 7],
            [175, 87, 57, 43, 34, 28, 24, 21, 18, 16, 15, 13, 12, 11, 10, 9, 9, 8, 8, 7],
            [181, 90, 59, 44, 35, 29, 25, 21, 19, 17, 15, 14, 13, 12, 11, 10, 10, 9, 8, 8],
            [187, 93, 61, 46, 36, 30, 25, 22, 19, 17, 16, 14, 13, 12, 11, 10, 10, 9, 8, 8],
            [191, 95, 63, 47, 37, 31, 26, 23, 20, 18, 16, 15, 13, 12, 11, 11, 10, 9, 9, 8]
        ])

        i1 = max(0, min(10, round(Q * 10)))
        i2 = max(0, min(20, round(f)))

        delay_compensation = a[i1-1][i2-1]

        # Get oscillator mass
        T = 1./f  # Oscillator period
        k = 1     # Fix spring constant k = 1, define only mass
        b = 2*np.pi/T
        m = k/b**2  # Mass of the oscillator

        # Get oscillator damping coefficient
        c = np.sqrt(m*k)/Q

        # if verbose:
        #     print('Oscillator Q-value: %0.4f, center frequency: %0.1f Hz, bandwidth: %0.1f Hz.\n' % (Q, 1/T, 1/T/Q))

        # Do zero padding
        e = np.transpose(ENVELOPE)
        e = np.vstack((e, np.zeros((500, e.shape[1]))))
        F = e.shape[1]  # Number of frequency channels

        # Get oscillator amplitudes as a function of time
        x = np.zeros((e.shape[0], F))
        a = np.zeros((e.shape[0], F))
        v = np.zeros((e.shape[0], F))

        for t in range(1, e.shape[0]):
            for cf in range(F):
                f_up = e[t, cf]  # driving positive force
                f_down = -k * x[t-1, cf] - c * v[t-1, cf]
                f_tot = f_up + f_down  # Total force
                # Get acceleration from force
                a[t, cf] = f_tot/m

                # Get velocity from acceleration
                v[t, cf] = v[t-1, cf] + a[t, cf] * 0.001  # assumes 1000 Hz sampling
                # Get position from velocity
                x[t, cf] = x[t-1, cf] + v[t, cf] * 0.001

        # Perform group delay correction by removing samples from the
        # beginning and adding zeroes to the end
        for f in range(F):
            if delay_compensation:
                x[:, f] = np.append(x[delay_compensation:, f], np.zeros((delay_compensation, 1)))

        x = x[:-500]  # Remove zero-padding

        # Combine N most energetic bands to get sonority envelope
        tmp = x
        tmp = tmp - np.min(tmp) + 0.00001
        x = np.zeros((tmp.shape[0], 1))

        for zz in range(tmp.shape[0]):
            sort_tmp = np.sort(tmp[zz, :], axis=0)[::-1]
            x[zz] = sum((np.log10(sort_tmp[:N])))

        # Scale sonority envelope between 0 and 1
        x = x - np.min(x)
        x = x / np.max(x)
        return x

# Generate Gammatone filterbank center frequencies (log-spacing)
    minfreq = 50
    maxfreq = 7500
    bands = 20

    cfs = np.zeros((bands, 1))
    const = (maxfreq/minfreq)**(1/(bands-1))

    cfs[0] = 50
    for k in range(bands-1):
        cfs[k+1] = cfs[k] * const

    # Read the audio data
    wav_data, fs = librosa.load(file_path)
    wav_data = librosa.resample(y=wav_data, orig_sr=fs, target_sr=16000)
    fs = 16000
    # Compute gammatone envelopes and downsample to 1000 Hz
    coefs = gammatone.filters.make_erb_filters(fs, cfs, width=1.0)
    filtered_signal = gammatone.filters.erb_filterbank(wav_data, coefs)
    hilbert_envelope = np.abs(hilbert(filtered_signal))
    env = librosa.resample(y=hilbert_envelope, orig_sr=fs, target_sr=1000)

    # Run oscillator-based segmentation
    Q_value = 0.5  # Q-value of the oscillator, default = 0.5 = critical damping
    center_frequency = 5  # in Hz
    threshold = 0.01

    # Get the sonority function
    outh = thetaOscillator(env, center_frequency, Q_value, threshold)

    # Detect the peaks and valleys of the sonority function
    peaks, valleys = peakdet(outh, threshold)
    syllable_timestamps = []

    if len(valleys) and len(peaks):
        valley_indices = valleys[:, 0]
        peak_indices = peaks[:, 0]

        # Add signal onset if not detected by valley picking
        if valley_indices[0] > 50:
            valley_indices = np.insert(valley_indices, 0, 0)
        if valley_indices[-1] < env.shape[1] - 50:
            valley_indices = np.append(valley_indices, env.shape[1])
    else:
        valley_indices = [0, len(env)]


    #Made changes here to retrienve the timestamps
    #-------------------------------------------------------------------------------
    # Collect the start and end times of each syllable
    for i in range(len(valley_indices) - 1):
        start_time = valley_indices[i] / 1000.0
        end_time = valley_indices[i + 1] / 1000.0
        syllable_timestamps.append((start_time, end_time))
    #-------------------------------------------------------------------------------


    # Plotting the results
    # pl.figure(figsize=(18, 8))
    # time_axis = [i * 1. / fs for i in range(len(wav_data))]
    # pl.plot(time_axis, wav_data)
    # pl.xlim((-0.01, (len(wav_data)) * 1. / fs + 0.01))
    # time_axis = [i * 1. / 1000 for i in range(len(outh))]
    # pl.plot(time_axis, outh)

    # for vi in valley_indices:
    #     pl.axvline(vi * 1. / 1000, ymin=0, ymax=1, color='r', linestyle='dashed')

    # pl.legend(['signal', 'sonority', 'estimated syll boundaries'])
    # pl.title('Theta oscillator based syllable boundary detection')
    # pl.xlabel('Time [s]')
    # pl.savefig('output.png')

    # Display the audio
    import IPython
    IPython.display.Audio(wav_data, rate=fs)

    # Print the syllable timestamps
    # print(syllable_timestamps)
    # for i, (start, end) in enumerate(syllable_timestamps):
    #     print(f"Syllable {i+1}: Start = {start:.3f} s, End = {end:.3f} s")
    return(syllable_timestamps)

def load_data_from_folders(folder_paths, syllable_boundaries_function):
    X = []
    y = []
    labels = {folder: idx for idx, folder in enumerate(folder_paths)}

    for folder in tqdm(folder_paths):
        for filename in os.listdir(folder):
            if filename.endswith (".wav"):
                file_path = os.path.join(folder, filename)
                syllable_boundaries = syllable_boundaries_function(file_path)  # Define this function
                mfcc_features = extract_mfcc_from_syllables(file_path, syllable_boundaries)
                #for m in mfcc_features:
                    #print(f"mfcc length ------------------------------------------{len(m)}")
                X.append(np.array(mfcc_features))
                y.append([labels[folder]])
    #print(f"X shape is {(len(X))} ,{(len(y))} , {(syllable_boundaries)},  type is {type(mfcc_features)} x is {X}")

    return X, y
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input
from tensorflow.keras.models import Model

def build_bilstm_feature_extractor(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(32, activation='relu')(x)  # Fixed-size feature representation
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
from tensorflow.keras.layers import TimeDistributed, Flatten

def build_tdnn_model(input_shape):
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        TimeDistributed(Dense(64, activation='relu')),
        TimeDistributed(Dense(32, activation='relu')),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
from sklearn.model_selection import train_test_split

# Define folder paths and syllable boundary extraction function
folder_paths = ['/home/spl_cair/Desktop/priyanka/icassp_exp/TISA_REP_OUT', '/home/spl_cair/Desktop/priyanka/icassp_exp/IED_REP_OUT']
#syllable_boundaries_function = lambda x: [(0, 1)]  # Define this function based on your syllable boundary extraction method

# Load data
X, y = load_data_from_folders(folder_paths, syllable_boundaries_function)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#X_train=np.array(X)

#print(f"type of input ---------------{type(X_train)}")

input_shape = (None, 13) 
bilstm_model = build_bilstm_feature_extractor(input_shape)
bilstm_model.summary()
#print(input_shape)

# Extract features
#input_data_with_batch = np.expand_dims(X_train, axis=0) 
#print(f" batch{input_data_with_batch.shape}, X_train {X_train.shape}")
X_train_features = bilstm_model.predict(X_train)
X_test_features = bilstm_model.predict(X_test)
# print(X_train.shape)

# Build and train TDNN model
tdnn_model = build_tdnn_model((X_train_features.shape[1], 32))  # Adjust input_shape based on BiLSTM output
tdnn_model.summary()
tdnn_model.fit(X_train_features, y_train, epochs=10, batch_size=1, validation_split=0.2)

# Evaluate the model
loss, accuracy = tdnn_model.evaluate(X_test_features, y_test)
print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
