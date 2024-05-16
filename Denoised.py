import os
import numpy as np
import librosa
import soundfile as sf

def spectral_subtraction(signal, sr, n_fft=1024, hop_length=512, win_length=1024, alpha=3, beta=0.002):
    """
    执行谱减法降噪

    参数:
    signal: 音频信号数组
    sr: 采样率
    n_fft: FFT窗口大小
    hop_length: 帧移
    win_length: 窗口长度
    alpha: 噪声衰减因子
    beta: 谱楼阈值

    返回:
    de_emphasized_signal: 降噪后的音频信号
    """
    
    signal = signal / signal.max()


    # 预加重处理增强高频
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # 短时傅里叶变换(STFT)
    S = librosa.stft(emphasized_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_magnitude, S_phase = np.abs(S), np.angle(S)

    # 估计噪声功率谱，使用前几帧进行估计
    noise_frames = 10
    noise_magnitude = np.mean(S_magnitude[:, :noise_frames], axis=1, keepdims=True)

    # 谱减法处理
    subtracted_magnitude = S_magnitude - alpha * noise_magnitude
    subtracted_magnitude = np.maximum(subtracted_magnitude, beta * S_magnitude)

    # 逆傅里叶变换恢复信号
    Y = subtracted_magnitude * np.exp(1j * S_phase)
    recovered_signal = librosa.istft(Y, hop_length=hop_length, win_length=win_length)

    # 反预加重处理
    de_emphasized_signal = np.append(recovered_signal[0], recovered_signal[1:] + pre_emphasis * recovered_signal[:-1])

    return de_emphasized_signal


def process_audio_files(root_dir):
    """
    遍历指定目录及子目录中的音频文件并进行降噪处理

    参数:
    root_dir: 包含音频文件的根目录
    """
    print(f"Scanning directory: {root_dir}")
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                print(f"Processing: {file_path}")

                signal, sr = librosa.load(file_path, sr=None)
                denoised_signal = spectral_subtraction(signal, sr)

                output_file_path = os.path.join(subdir, 'denoised_' + file)
                sf.write(output_file_path, denoised_signal, sr)  # 使用 soundfile 保存文件
                print(f"Processed and saved: {output_file_path}")


# 指定音频文件所在的根目录
root_directory = r'C:\Users\12430\Desktop\ee\fyp\DCASEFoleySoundSynthesisDevSet'
process_audio_files(root_directory)

##先做规划
