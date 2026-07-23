
import numpy as np
import torch
from search_config_2d import N_FFT, HOP_LENGTH, FREQ_BINS, FS


def waveform_to_spectrogram(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, freq_bins=FREQ_BINS):
    """
    将单条波形转换为对数幅度语谱图。
    
    返回形状：(freq_bins, n_frames)，其中 n_frames 由 STFT 自然确定。
    不再对时间维度进行截断或填充，保留完整的时频信息。
    """
    import librosa
    
    # STFT 变换
    stft_complex = librosa.stft(
        waveform.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
        window='hann'
    )
    
    # 对数幅度转换
    magnitude = np.abs(stft_complex)
    eps = 1e-8
    log_magnitude = 20.0 * np.log10(magnitude + eps)
    
    # 仅截断频率维度（前 freq_bins 个bin），时间维度保持自然长度
    log_magnitude = log_magnitude[:freq_bins, :]
    
    # Min-Max 归一化到 [0, 1]
    spec_min = log_magnitude.min()
    spec_max = log_magnitude.max()
    denom = spec_max - spec_min
    if denom < 1e-8:
        normalized = np.zeros_like(log_magnitude)
    else:
        normalized = (log_magnitude - spec_min) / denom
    
    return normalized.astype(np.float32)


def batch_to_spectrograms(waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH, freq_bins=FREQ_BINS):
    """
    批量转换为语谱图张量。
    
    返回形状：(B, 1, freq_bins, n_frames)，其中 n_frames 由首个样本确定。
    因为所有样本窗长相同、STFT 参数相同，帧数一致。
    """
    if isinstance(waveforms, torch.Tensor):
        waveforms = waveforms.cpu().numpy()
    if waveforms.ndim == 3:
        waveforms = waveforms.squeeze(1)
    
    batch_size = waveforms.shape[0]
    
    # 先用首个样本确定时间维度
    first_spec = waveform_to_spectrogram(waveforms[0], n_fft=n_fft, hop_length=hop_length, freq_bins=freq_bins)
    n_frames = first_spec.shape[1]
    
    # 分配批量张量
    specs = np.zeros((batch_size, 1, freq_bins, n_frames), dtype=np.float32)
    specs[0, 0] = first_spec
    
    for i in range(1, batch_size):
        spec = waveform_to_spectrogram(waveforms[i], n_fft=n_fft, hop_length=hop_length, freq_bins=freq_bins)
        specs[i, 0] = spec
    
    return torch.from_numpy(specs)


if __name__ == '__main__':
    pass