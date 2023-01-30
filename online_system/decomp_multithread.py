import threading
import time
import sys
import socket
import numpy as np
from scipy import signal
import numpy as np
import pandas as pd
import pickle, json

import serial

path_model = '/'

# load FastIca model
with open(path_model + 'all_fastica.pickle', 'rb') as f:
    emg_FastICA1 = pickle.load(f)

# load kmeans model
list_kmeans = []
for j in range(20):
    with open(path_model + f'kmeans_{str(j)}.pickle', 'rb') as f:
        list_kmeans.append(pickle.load(f))

with open(path_model + f'all_sil.json', 'rb') as f:
    d = json.load(f)
valid_index_mu = d['valid_index_mu']

with open(path_model + 'svc.pickle', 'rb') as f:
    svm_clf = pickle.load(f)


sos = signal.butter(16, [5, 150], analog=False, btype='band', output='sos', fs=2000)
b1, a1 = signal.iirnotch(50, 50, 2000)
b2, a2 = signal.iirnotch(100, 10, 2000)
b3, a3 = signal.iirnotch(150, 15, 2000)
b4, a4 = signal.iirnotch(200, 20, 2000)

def guolv(x):
    y = signal.sosfilt(sos, x)
    
    y1 = signal.filtfilt(b1, a1, y)
    y2 = signal.filtfilt(b2, a2, y1)
    y3 = signal.filtfilt(b3, a3, y2)
    y4 = signal.filtfilt(b4, a4, y3)

    return y4

def extend_data(emg_raw):
    df_emg_raw = pd.DataFrame(emg_raw)
    emg_extended = pd.concat([df_emg_raw] + [df_emg_raw.shift(-x) for x in range(8)], axis=1).dropna()
    emg_centered = emg_extended - np.mean(emg_extended, axis=0)
    return emg_centered

def readUint32(array, arrayIndex):
    variableBytes = array[arrayIndex : arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex = arrayIndex + 4
    return variable, arrayIndex

def readInt32(array, arrayIndex):
    variableBytes = array[arrayIndex : arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=True)
    arrayIndex = arrayIndex + 4
    return variable, arrayIndex

def readUint16(array, arrayIndex):
    variableBytes = array[arrayIndex : arrayIndex + 2]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex = arrayIndex + 2
    return variable, arrayIndex

class RealtimeEmgProcessor():
    def __init__(self, channel_names: list, numBlocks: int, commands=None, com_number=None) -> None:
        self.event = threading.Event()
        self.scommand, self.swaveform, self.timestep = self._init_rhd(channel_names)
        
        self.sos = self._init_filter_bandpass()
        self.comb_b1, self.comb_a1 = self._init_filter_comb1()
        self.comb_b2, self.comb_a2 = self._init_filter_comb2()
        self.comb_b3, self.comb_a3 = self._init_filter_comb3()
        self.comb_b4, self.comb_a4 = self._init_filter_comb4()
        
        self.channel_names = channel_names
        self.numBlocks = numBlocks

        self.blocksAmplifierData = []

        if com_number:
            self.commands = commands
            self.ser = serial.Serial('COM' + str(com_number), 9600)
        else:
            self.ser = None
    
    def _init_rhd(self, channel_names: list):
        # Declare buffer size for reading from TCP command socket
        # This is the maximum number of bytes expected for 1 read. 1024 is plenty for a single text command
        COMMAND_BUFFER_SIZE = 1024 # Increase if many return commands are expected

        # Connect to TCP command server - default home IP address at port 5000
        print('Connecting to TCP command server...')
        scommand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        scommand.connect(('127.0.0.1', 5000))

        # Connect to TCP waveform server - default home IP address at port 5001
        print('Connecting to TCP waveform server...')
        swaveform = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        swaveform.connect(('127.0.0.1', 5001))

        # Query runmode from RHX software
        scommand.sendall(b'get runmode')
        commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
        isStopped = commandReturn == "Return: RunMode Stop"

        # If controller is running, stop it
        if not isStopped:
            scommand.sendall(b'set runmode stop')
            time.sleep(0.1) # Allow time for RHX software to accept this command before the next one comes

        # Query sample rate from RHX software
        scommand.sendall(b'get sampleratehertz')
        commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
        expectedReturnString = "Return: SampleRateHertz "
        if commandReturn.find(expectedReturnString) == -1: # Look for "Return: SampleRateHertz N" where N is the sample rate
            raise Exception('Unable to get sample rate from server')
        else:
            sampleRate = float(commandReturn[len(expectedReturnString):])

        # Calculate timestep from sample rate
        timestep = 1 / sampleRate

        # Clear TCP data output to ensure no TCP channels are enabled
        scommand.sendall(b'execute clearalldataoutputs')
        time.sleep(0.1)

        # Send TCP commands to set up TCP Data Output Enabled for wide
        for channel_name in channel_names:
            for i in range(64):
                channel_index = str(i).rjust(3, '0')
                _command = f'set {channel_name}-{channel_index}.tcpdataoutputenabled true'
                scommand.sendall(_command.encode('utf-8'))
                time.sleep(0.1)

        return scommand, swaveform, timestep
    
    def emg_getter(self):
        # emg配列が用意されていない
        self.event.clear()

        # Calculations for accurate parsing
        # At 30 kHz with 1 channel, 1 second of wideband waveform data (including magic number, timestamps, and amplifier data) is 181,420 bytes
        # N = (framesPerBlock * waveformBytesPerFrame + SizeOfMagicNumber) * NumBlocks where:
        # framesPerBlock = 128 ; standard data block size used by Intan
        # waveformBytesPerFrame = SizeOfTimestamp + SizeOfSample ; timestamp is a 4-byte (32-bit) int, and amplifier sample is a 2-byte (16-bit) unsigned int
        # SizeOfMagicNumber = 4; Magic number is a 4-byte (32-bit) unsigned int
        # NumBlocks = NumFrames / framesPerBlock ; At 30 kHz, 1 second of data has 30000 frames. NumBlocks must be an integer value, so round up to 235

        numAmpChannels = len(self.channel_names) * 64
        framesPerBlock = 128
        waveformBytesPerFrame = 4 + 2 * numAmpChannels
        waveformBytesPerBlock = framesPerBlock * waveformBytesPerFrame + 4
        waveformBytesPerBlocks = self.numBlocks * waveformBytesPerBlock

        # Run controller
        self.scommand.sendall(b'set runmode run')
        time.sleep(0.1)
        
        while True:
            # Read waveform data
            rawData = self.swaveform.recv(waveformBytesPerBlocks)
            if len(rawData) % waveformBytesPerBlock != 0:
                raise Exception('An unexpected amount of data arrived that is not an integer multiple of the expected data size per block')

            if len(rawData) != waveformBytesPerBlocks:
                continue
            
            # 配列をリセットするのでフラグをおろす
            self.event.clear()

            self.blocksAmplifierData = [] # List used to contain scaled amplifier data
            rawIndex = 0 # Index used to read the raw data that came in through the TCP socket

            for block in range(self.numBlocks):
                # Expect 4 bytes to be TCP Magic Number as uint32.
                # If not what's expected, raise an exception.
                magicNumber, rawIndex = readUint32(rawData, rawIndex)
                if magicNumber != 0x2ef07a08:
                    raise Exception('Error... magic number incorrect')

                # Each block should contain 128 frames of data - process each
                # of these one-by-one
                for frame in range(framesPerBlock):
                    amplifierData = []
                    # Expect 4 bytes to be timestamp as int32.
                    rawTimestamp, rawIndex = readInt32(rawData, rawIndex)
                    
                    # Multiply by 'timestep' to convert timestamp to seconds
                    #if frame == 0:
                    #    print(rawTimestamp * self.timestep)
                    #amplifierTimestamps.append(rawTimestamp * self.timestep)

                    for num_channel in range(numAmpChannels):
                        # Expect 2 bytes of wideband data.
                        rawSample, rawIndex = readUint16(rawData, rawIndex)
                        
                        amplifierData.append(rawSample)
                    self.blocksAmplifierData.append(amplifierData)
            
            # emg配列が用意されたフラグを立てる
            self.event.set()
    
    def emg_processor(self):
        #s = time.time()
        #old_data=np.zeros((377,128))
        while True:
            if self.event.is_set():  # 配列が用意されているか確認
                # Scale this sample to convert to microVolts
                raw_emg = 0.195 * (np.array(self.blocksAmplifierData) - 32768)
                
                filtered_emg = guolv(raw_emg.T)
                processing_data = filtered_emg.T
                #processing_data = np.concatenate((old_data, filtered_emg),0)
                #print(processing_data.shape)
                #old_data = filtered_emg
                #time.sleep(0.1)
                
                # extend
                emg_raw1 = processing_data[:, 0:64]
                
                emg_extend1 = extend_data(emg_raw1)
                
                emg_mu1 = emg_FastICA1.transform(emg_extend1)
                emg_mu_squared1 = np.square(emg_mu1)
                spike_trains1 = np.zeros_like(emg_mu_squared1)
                
                # decomposition
                for i in range(emg_mu_squared1.shape[1]):
                    _kmeans = list_kmeans[i]
                    idx = np.argsort(_kmeans.cluster_centers_.sum(axis=1))
                    flag = np.zeros_like(idx)
                    flag[idx] = np.arange(len(idx))
                    
                    spike_trains1[:, i] = flag[np.argmin(_kmeans.transform(emg_mu_squared1[:, [i]]), axis=1)]

                pre_diff = pd.DataFrame(emg_mu_squared1).diff(-1) > 0
                post_diff = pd.DataFrame(emg_mu_squared1).diff(1) > 0
                spike_trains_processsed1 = spike_trains1 * pre_diff.values * post_diff.values
                
                # spike_trains = np.concatenate((spike_trains_processsed1[:, 1:10], spike_trains_processsed2[:, 0:9]),1)

                fr_new = np.sum(spike_trains_processsed1, axis=0)[:, valid_index_mu]
                
                #print(fr_new)
                # classification
                label_predicted = int(svm_clf.transform(fr_new)[0])
                print(label_predicted)
                if self.ser:
                    command = self.commands[label_predicted]
                    self.ser.write(str(label_predicted))
                #print(time.time()-s) 
                
            else:
                self.event.wait()   # flag=Trueになるまでここでブロッキングする
    
    def main_thread(self):
        while True:
            try:
                _ = input()
            except KeyboardInterrupt:
                self.scommand.sendall(b'set runmode stop')
                self.ser.close()
                time.sleep(0.1)
                print('end processor...')
                sys.exit()

if __name__ == '__main__':
    realtime_emg_processor = RealtimeEmgProcessor(['b'], 6)
    # realtime_emg_processor = RealtimeEmgProcessor(['b'], 6, commands=['r', 'a', 'b', 'c', 'd', 'e'], com_number=3)
    
    emg_getter = threading.Thread(target=realtime_emg_processor.emg_getter,)
    emg_getter.setDaemon(True)
    emg_getter.start()

    emg_processor = threading.Thread(target=realtime_emg_processor.emg_processor)
    emg_processor.setDaemon(True)
    emg_processor.start()

    realtime_emg_processor.main_thread()
