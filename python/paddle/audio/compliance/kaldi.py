# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from torchaudio(https://github.com/pytorch/audio)

import math
from typing import Tuple

import paddle
from paddle import Tensor

from ..functional import create_dct
from ..functional.window import get_window

__all__ = [
    'spectrogram',
    'fbank',
    'mfcc',
]

# window types
HANNING = 'hann'
HAMMING = 'hamming'
POVEY = 'povey'
RECTANGULAR = 'rect'
BLACKMAN = 'blackman'


def _get_epsilon(dtype):
    return paddle.to_tensor(1e-07, dtype=dtype)


def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2**(x - 1).bit_length()


def _get_strided(waveform: Tensor,
                 window_size: int,
                 window_shift: int,
                 snip_edges: bool) -> Tensor:
    assert waveform.dim() == 1
    num_samples = waveform.shape[0]

    if snip_edges:
        if num_samples < window_size:
            return paddle.empty((0, 0), dtype=waveform.dtype)
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = paddle.flip(waveform, [0])
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0:
            pad_left = reversed_waveform[-pad:]
            waveform = paddle.concat((pad_left, waveform, pad_right), axis=0)
        else:
            waveform = paddle.concat((waveform[-pad:], pad_right), axis=0)

    return paddle.signal.frame(waveform, window_size, window_shift)[:, :m].T


def _feature_window_function(
        window_type: str,
        window_size: int,
        blackman_coeff: float,
        dtype: int, ) -> Tensor:
    if window_type == HANNING:
        return get_window('hann', window_size, fftbins=False, dtype=dtype)
    elif window_type == HAMMING:
        return get_window('hamming', window_size, fftbins=False, dtype=dtype)
    elif window_type == POVEY:
        return get_window(
            'hann', window_size, fftbins=False, dtype=dtype).pow(0.85)
    elif window_type == RECTANGULAR:
        return paddle.ones([window_size], dtype=dtype)
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        window_function = paddle.arange(window_size, dtype=dtype)
        return (blackman_coeff - 0.5 * paddle.cos(a * window_function) +
                (0.5 - blackman_coeff) * paddle.cos(2 * a * window_function)
                ).astype(dtype)
    else:
        raise Exception('Invalid window type ' + window_type)


def _get_log_energy(strided_input: Tensor, epsilon: Tensor,
                    energy_floor: float) -> Tensor:
    log_energy = paddle.maximum(strided_input.pow(2).sum(1), epsilon).log()
    if energy_floor == 0.0:
        return log_energy
    return paddle.maximum(
        log_energy,
        paddle.to_tensor(math.log(energy_floor), dtype=strided_input.dtype))


def _get_waveform_and_window_properties(
        waveform: Tensor,
        channel: int,
        sr: int,
        frame_shift: float,
        frame_length: float,
        round_to_power_of_two: bool,
        preemphasis_coefficient: float) -> Tuple[Tensor, int, int, int]:
    channel = max(channel, 0)
    assert channel < waveform.shape[0], (
        'Invalid channel {} for size {}'.format(channel, waveform.shape[0]))
    waveform = waveform[channel, :]  # size (n)
    window_shift = int(
        sr * frame_shift *
        0.001)  # pass frame_shift and frame_length in milliseconds
    window_size = int(sr * frame_length * 0.001)
    padded_window_size = _next_power_of_2(
        window_size) if round_to_power_of_two else window_size

    assert 2 <= window_size <= len(waveform), (
        'choose a window size {} that is [2, {}]'.format(window_size,
                                                         len(waveform)))
    assert 0 < window_shift, '`window_shift` must be greater than 0'
    assert padded_window_size % 2 == 0, 'the padded `window_size` must be divisible by two.' \
                                        ' use `round_to_power_of_two` or change `frame_length`'
    assert 0. <= preemphasis_coefficient <= 1.0, '`preemphasis_coefficient` must be between [0,1]'
    assert sr > 0, '`sr` must be greater than zero'
    return waveform, window_shift, window_size, padded_window_size


def _get_window(waveform: Tensor,
                padded_window_size: int,
                window_size: int,
                window_shift: int,
                window_type: str,
                blackman_coeff: float,
                snip_edges: bool,
                raw_energy: bool,
                energy_floor: float,
                dither: float,
                remove_dc_offset: bool,
                preemphasis_coefficient: float) -> Tuple[Tensor, Tensor]:
    dtype = waveform.dtype
    epsilon = _get_epsilon(dtype)

    # (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift,
                                 snip_edges)

    if dither != 0.0:
        x = paddle.maximum(epsilon,
                           paddle.rand(strided_input.shape, dtype=dtype))
        rand_gauss = paddle.sqrt(-2 * x.log()) * paddle.cos(2 * math.pi * x)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        row_means = paddle.mean(strided_input, axis=1).unsqueeze(1)  # (m, 1)
        strided_input = strided_input - row_means

    if raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon,
                                            energy_floor)  # (m)

    if preemphasis_coefficient != 0.0:
        offset_strided_input = paddle.nn.functional.pad(
            strided_input.unsqueeze(0), (1, 0),
            data_format='NCL',
            mode='replicate').squeeze(0)  # (m, window_size + 1)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :
                                                                                       -1]

    window_function = _feature_window_function(
        window_type, window_size, blackman_coeff,
        dtype).unsqueeze(0)  # (1, window_size)
    strided_input = strided_input * window_function  # (m, window_size)

    # (m, padded_window_size)
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = paddle.nn.functional.pad(
            strided_input.unsqueeze(0), (0, padding_right),
            data_format='NCL',
            mode='constant',
            value=0).squeeze(0)

    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon,
                                            energy_floor)  # size (m)

    return strided_input, signal_log_energy


def _subtract_column_mean(tensor: Tensor, subtract_mean: bool) -> Tensor:
    if subtract_mean:
        col_means = paddle.mean(tensor, axis=0).unsqueeze(0)
        tensor = tensor - col_means
    return tensor


def spectrogram(waveform: Tensor,
                blackman_coeff: float=0.42,
                channel: int=-1,
                dither: float=0.0,
                energy_floor: float=1.0,
                frame_length: float=25.0,
                frame_shift: float=10.0,
                preemphasis_coefficient: float=0.97,
                raw_energy: bool=True,
                remove_dc_offset: bool=True,
                round_to_power_of_two: bool=True,
                sr: int=16000,
                snip_edges: bool=True,
                subtract_mean: bool=False,
                window_type: str=POVEY) -> Tensor:
    """Compute and return a spectrogram from a waveform. The output is identical to Kaldi's.

    Args:
        waveform (Tensor): A waveform tensor with shape `(C, T)`.
        blackman_coeff (float, optional): Coefficient for Blackman window.. Defaults to 0.42.
        channel (int, optional): Select the channel of waveform. Defaults to -1.
        dither (float, optional): Dithering constant . Defaults to 0.0.
        energy_floor (float, optional): Floor on energy of the output Spectrogram. Defaults to 1.0.
        frame_length (float, optional): Frame length in milliseconds. Defaults to 25.0.
        frame_shift (float, optional): Shift between adjacent frames in milliseconds. Defaults to 10.0.
        preemphasis_coefficient (float, optional): Preemphasis coefficient for input waveform. Defaults to 0.97.
        raw_energy (bool, optional): Whether to compute before preemphasis and windowing. Defaults to True.
        remove_dc_offset (bool, optional): Whether to subtract mean from waveform on frames. Defaults to True.
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. Defaults to True.
        sr (int, optional): Sample rate of input waveform. Defaults to 16000.
        snip_edges (bool, optional): Drop samples in the end of waveform that cann't fit a singal frame when it
            is set True. Otherwise performs reflect padding to the end of waveform. Defaults to True.
        subtract_mean (bool, optional): Whether to subtract mean of feature files. Defaults to False.
        window_type (str, optional): Choose type of window for FFT computation. Defaults to POVEY.

    Returns:
        Tensor: A spectrogram tensor with shape `(m, padded_window_size // 2 + 1)` where m is the number of frames
            depends on frame_length and frame_shift.
    """
    dtype = waveform.dtype
    epsilon = _get_epsilon(dtype)

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sr, frame_shift, frame_length, round_to_power_of_two,
        preemphasis_coefficient)

    strided_input, signal_log_energy = _get_window(
        waveform, padded_window_size, window_size, window_shift, window_type,
        blackman_coeff, snip_edges, raw_energy, energy_floor, dither,
        remove_dc_offset, preemphasis_coefficient)

    # (m, padded_window_size // 2 + 1, 2)
    fft = paddle.fft.rfft(strided_input)

    power_spectrum = paddle.maximum(
        fft.abs().pow(2.), epsilon).log()  # (m, padded_window_size // 2 + 1)
    power_spectrum[:, 0] = signal_log_energy

    power_spectrum = _subtract_column_mean(power_spectrum, subtract_mean)
    return power_spectrum


def _inverse_mel_scale_scalar(mel_freq: float) -> float:
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


def _inverse_mel_scale(mel_freq: Tensor) -> Tensor:
    return 700.0 * ((mel_freq / 1127.0).exp() - 1.0)


def _mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def _mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


def _vtln_warp_freq(vtln_low_cutoff: float,
                    vtln_high_cutoff: float,
                    low_freq: float,
                    high_freq: float,
                    vtln_warp_factor: float,
                    freq: Tensor) -> Tensor:
    assert vtln_low_cutoff > low_freq, 'be sure to set the vtln_low option higher than low_freq'
    assert vtln_high_cutoff < high_freq, 'be sure to set the vtln_high option lower than high_freq [or negative]'
    l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l
    Fh = scale * h
    assert l > low_freq and h < high_freq
    scale_left = (Fl - low_freq) / (l - low_freq)
    scale_right = (high_freq - Fh) / (high_freq - h)
    res = paddle.empty_like(freq)

    outside_low_high_freq = paddle.less_than(freq, paddle.to_tensor(low_freq)) \
        | paddle.greater_than(freq, paddle.to_tensor(high_freq))
    before_l = paddle.less_than(freq, paddle.to_tensor(l))
    before_h = paddle.less_than(freq, paddle.to_tensor(h))
    after_h = paddle.greater_equal(freq, paddle.to_tensor(h))

    res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
    res[before_h] = scale * freq[before_h]
    res[before_l] = low_freq + scale_left * (freq[before_l] - low_freq)
    res[outside_low_high_freq] = freq[outside_low_high_freq]

    return res


def _vtln_warp_mel_freq(vtln_low_cutoff: float,
                        vtln_high_cutoff: float,
                        low_freq,
                        high_freq: float,
                        vtln_warp_factor: float,
                        mel_freq: Tensor) -> Tensor:
    return _mel_scale(
        _vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq,
                        vtln_warp_factor, _inverse_mel_scale(mel_freq)))


def _get_mel_banks(num_bins: int,
                   window_length_padded: int,
                   sample_freq: float,
                   low_freq: float,
                   high_freq: float,
                   vtln_low: float,
                   vtln_high: float,
                   vtln_warp_factor: float) -> Tuple[Tensor, Tensor]:
    assert num_bins > 3, 'Must have at least 3 mel bins'
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq), \
        ('Bad values in options: low-freq {} and high-freq {} vs. nyquist {}'.format(low_freq, high_freq, nyquist))

    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = _mel_scale_scalar(low_freq)
    mel_high_freq = _mel_scale_scalar(high_freq)

    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    if vtln_high < 0.0:
        vtln_high += nyquist

    assert vtln_warp_factor == 1.0 or ((low_freq < vtln_low < high_freq) and
                                       (0.0 < vtln_high < high_freq) and (vtln_low < vtln_high)), \
        ('Bad values in options: vtln-low {} and vtln-high {}, versus '
         'low-freq {} and high-freq {}'.format(vtln_low, vtln_high, low_freq, high_freq))

    bin = paddle.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin * mel_freq_delta  # (num_bins, 1)
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta  # (num_bins, 1)
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta  # (num_bins, 1)

    if vtln_warp_factor != 1.0:
        left_mel = _vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq,
                                       vtln_warp_factor, left_mel)
        center_mel = _vtln_warp_mel_freq(vtln_low, vtln_high, low_freq,
                                         high_freq, vtln_warp_factor,
                                         center_mel)
        right_mel = _vtln_warp_mel_freq(vtln_low, vtln_high, low_freq,
                                        high_freq, vtln_warp_factor, right_mel)

    center_freqs = _inverse_mel_scale(center_mel)  # (num_bins)
    # (1, num_fft_bins)
    mel = _mel_scale(fft_bin_width * paddle.arange(num_fft_bins)).unsqueeze(0)

    # (num_bins, num_fft_bins)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    if vtln_warp_factor == 1.0:
        bins = paddle.maximum(
            paddle.zeros([1]), paddle.minimum(up_slope, down_slope))
    else:
        bins = paddle.zeros_like(up_slope)
        up_idx = paddle.greater_than(mel, left_mel) & paddle.less_than(
            mel, center_mel)
        down_idx = paddle.greater_than(mel, center_mel) & paddle.less_than(
            mel, right_mel)
        bins[up_idx] = up_slope[up_idx]
        bins[down_idx] = down_slope[down_idx]

    return bins, center_freqs


def fbank(waveform: Tensor,
          blackman_coeff: float=0.42,
          channel: int=-1,
          dither: float=0.0,
          energy_floor: float=1.0,
          frame_length: float=25.0,
          frame_shift: float=10.0,
          high_freq: float=0.0,
          htk_compat: bool=False,
          low_freq: float=20.0,
          n_mels: int=23,
          preemphasis_coefficient: float=0.97,
          raw_energy: bool=True,
          remove_dc_offset: bool=True,
          round_to_power_of_two: bool=True,
          sr: int=16000,
          snip_edges: bool=True,
          subtract_mean: bool=False,
          use_energy: bool=False,
          use_log_fbank: bool=True,
          use_power: bool=True,
          vtln_high: float=-500.0,
          vtln_low: float=100.0,
          vtln_warp: float=1.0,
          window_type: str=POVEY) -> Tensor:
    """Compute and return filter banks from a waveform. The output is identical to Kaldi's.

    Args:
        waveform (Tensor): A waveform tensor with shape `(C, T)`.
        blackman_coeff (float, optional): Coefficient for Blackman window.. Defaults to 0.42.
        channel (int, optional): Select the channel of waveform. Defaults to -1.
        dither (float, optional): Dithering constant . Defaults to 0.0.
        energy_floor (float, optional): Floor on energy of the output Spectrogram. Defaults to 1.0.
        frame_length (float, optional): Frame length in milliseconds. Defaults to 25.0.
        frame_shift (float, optional): Shift between adjacent frames in milliseconds. Defaults to 10.0.
        high_freq (float, optional): The upper cut-off frequency. Defaults to 0.0.
        htk_compat (bool, optional): Put energy to the last when it is set True. Defaults to False.
        low_freq (float, optional): The lower cut-off frequency. Defaults to 20.0.
        n_mels (int, optional): Number of output mel bins. Defaults to 23.
        preemphasis_coefficient (float, optional): Preemphasis coefficient for input waveform. Defaults to 0.97.
        raw_energy (bool, optional): Whether to compute before preemphasis and windowing. Defaults to True.
        remove_dc_offset (bool, optional): Whether to subtract mean from waveform on frames. Defaults to True.
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. Defaults to True.
        sr (int, optional): Sample rate of input waveform. Defaults to 16000.
        snip_edges (bool, optional): Drop samples in the end of waveform that cann't fit a singal frame when it
            is set True. Otherwise performs reflect padding to the end of waveform. Defaults to True.
        subtract_mean (bool, optional): Whether to subtract mean of feature files. Defaults to False.
        use_energy (bool, optional): Add an dimension with energy of spectrogram to the output. Defaults to False.
        use_log_fbank (bool, optional): Return log fbank when it is set True. Defaults to True.
        use_power (bool, optional): Whether to use power instead of magnitude. Defaults to True.
        vtln_high (float, optional): High inflection point in piecewise linear VTLN warping function. Defaults to -500.0.
        vtln_low (float, optional): Low inflection point in piecewise linear VTLN warping function. Defaults to 100.0.
        vtln_warp (float, optional): Vtln warp factor. Defaults to 1.0.
        window_type (str, optional): Choose type of window for FFT computation. Defaults to POVEY.

    Returns:
        Tensor: A filter banks tensor with shape `(m, n_mels)`.
    """
    dtype = waveform.dtype

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sr, frame_shift, frame_length, round_to_power_of_two,
        preemphasis_coefficient)

    strided_input, signal_log_energy = _get_window(
        waveform, padded_window_size, window_size, window_shift, window_type,
        blackman_coeff, snip_edges, raw_energy, energy_floor, dither,
        remove_dc_offset, preemphasis_coefficient)

    # (m, padded_window_size // 2 + 1)
    spectrum = paddle.fft.rfft(strided_input).abs()
    if use_power:
        spectrum = spectrum.pow(2.)

    # (n_mels, padded_window_size // 2)
    mel_energies, _ = _get_mel_banks(n_mels, padded_window_size, sr, low_freq,
                                     high_freq, vtln_low, vtln_high, vtln_warp)
    mel_energies = mel_energies.astype(dtype)

    # (n_mels, padded_window_size // 2 + 1)
    mel_energies = paddle.nn.functional.pad(
        mel_energies.unsqueeze(0), (0, 1),
        data_format='NCL',
        mode='constant',
        value=0).squeeze(0)

    # (m, n_mels)
    mel_energies = paddle.mm(spectrum, mel_energies.T)
    if use_log_fbank:
        mel_energies = paddle.maximum(mel_energies, _get_epsilon(dtype)).log()

    if use_energy:
        signal_log_energy = signal_log_energy.unsqueeze(1)
        if htk_compat:
            mel_energies = paddle.concat(
                (mel_energies, signal_log_energy), axis=1)
        else:
            mel_energies = paddle.concat(
                (signal_log_energy, mel_energies), axis=1)

    # (m, n_mels + 1)
    mel_energies = _subtract_column_mean(mel_energies, subtract_mean)
    return mel_energies


def _get_dct_matrix(n_mfcc: int, n_mels: int) -> Tensor:
    dct_matrix = create_dct(n_mels, n_mels, 'ortho')
    dct_matrix[:, 0] = math.sqrt(1 / float(n_mels))
    dct_matrix = dct_matrix[:, :n_mfcc]  # (n_mels, n_mfcc)
    return dct_matrix


def _get_lifter_coeffs(n_mfcc: int, cepstral_lifter: float) -> Tensor:
    i = paddle.arange(n_mfcc)
    return 1.0 + 0.5 * cepstral_lifter * paddle.sin(math.pi * i /
                                                    cepstral_lifter)


def mfcc(waveform: Tensor,
         blackman_coeff: float=0.42,
         cepstral_lifter: float=22.0,
         channel: int=-1,
         dither: float=0.0,
         energy_floor: float=1.0,
         frame_length: float=25.0,
         frame_shift: float=10.0,
         high_freq: float=0.0,
         htk_compat: bool=False,
         low_freq: float=20.0,
         n_mfcc: int=13,
         n_mels: int=23,
         preemphasis_coefficient: float=0.97,
         raw_energy: bool=True,
         remove_dc_offset: bool=True,
         round_to_power_of_two: bool=True,
         sr: int=16000,
         snip_edges: bool=True,
         subtract_mean: bool=False,
         use_energy: bool=False,
         vtln_high: float=-500.0,
         vtln_low: float=100.0,
         vtln_warp: float=1.0,
         window_type: str=POVEY) -> Tensor:
    """Compute and return mel frequency cepstral coefficients from a waveform. The output is
            identical to Kaldi's.

    Args:
        waveform (Tensor): A waveform tensor with shape `(C, T)`.
        blackman_coeff (float, optional): Coefficient for Blackman window.. Defaults to 0.42.
        cepstral_lifter (float, optional): Scaling of output mfccs. Defaults to 22.0.
        channel (int, optional): Select the channel of waveform. Defaults to -1.
        dither (float, optional): Dithering constant . Defaults to 0.0.
        energy_floor (float, optional): Floor on energy of the output Spectrogram. Defaults to 1.0.
        frame_length (float, optional): Frame length in milliseconds. Defaults to 25.0.
        frame_shift (float, optional): Shift between adjacent frames in milliseconds. Defaults to 10.0.
        high_freq (float, optional): The upper cut-off frequency. Defaults to 0.0.
        htk_compat (bool, optional): Put energy to the last when it is set True. Defaults to False.
        low_freq (float, optional): The lower cut-off frequency. Defaults to 20.0.
        n_mfcc (int, optional): Number of cepstra in MFCC. Defaults to 13.
        n_mels (int, optional): Number of output mel bins. Defaults to 23.
        preemphasis_coefficient (float, optional): Preemphasis coefficient for input waveform. Defaults to 0.97.
        raw_energy (bool, optional): Whether to compute before preemphasis and windowing. Defaults to True.
        remove_dc_offset (bool, optional): Whether to subtract mean from waveform on frames. Defaults to True.
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. Defaults to True.
        sr (int, optional): Sample rate of input waveform. Defaults to 16000.
        snip_edges (bool, optional): Drop samples in the end of waveform that cann't fit a singal frame when it
            is set True. Otherwise performs reflect padding to the end of waveform. Defaults to True.
        subtract_mean (bool, optional): Whether to subtract mean of feature files. Defaults to False.
        use_energy (bool, optional): Add an dimension with energy of spectrogram to the output. Defaults to False.
        vtln_high (float, optional): High inflection point in piecewise linear VTLN warping function. Defaults to -500.0.
        vtln_low (float, optional): Low inflection point in piecewise linear VTLN warping function. Defaults to 100.0.
        vtln_warp (float, optional): Vtln warp factor. Defaults to 1.0.
        window_type (str, optional): Choose type of window for FFT computation. Defaults to POVEY.

    Returns:
        Tensor: A mel frequency cepstral coefficients tensor with shape `(m, n_mfcc)`.
    """
    assert n_mfcc <= n_mels, 'n_mfcc cannot be larger than n_mels: %d vs %d' % (
        n_mfcc, n_mels)

    dtype = waveform.dtype

    # (m, n_mels + use_energy)
    feature = fbank(
        waveform=waveform,
        blackman_coeff=blackman_coeff,
        channel=channel,
        dither=dither,
        energy_floor=energy_floor,
        frame_length=frame_length,
        frame_shift=frame_shift,
        high_freq=high_freq,
        htk_compat=htk_compat,
        low_freq=low_freq,
        n_mels=n_mels,
        preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy,
        remove_dc_offset=remove_dc_offset,
        round_to_power_of_two=round_to_power_of_two,
        sr=sr,
        snip_edges=snip_edges,
        subtract_mean=False,
        use_energy=use_energy,
        use_log_fbank=True,
        use_power=True,
        vtln_high=vtln_high,
        vtln_low=vtln_low,
        vtln_warp=vtln_warp,
        window_type=window_type)

    if use_energy:
        # (m)
        signal_log_energy = feature[:, n_mels if htk_compat else 0]
        mel_offset = int(not htk_compat)
        feature = feature[:, mel_offset:(n_mels + mel_offset)]

    # (n_mels, n_mfcc)
    dct_matrix = _get_dct_matrix(n_mfcc, n_mels).astype(dtype=dtype)

    # (m, n_mfcc)
    feature = feature.matmul(dct_matrix)

    if cepstral_lifter != 0.0:
        # (1, n_mfcc)
        lifter_coeffs = _get_lifter_coeffs(n_mfcc, cepstral_lifter).unsqueeze(0)
        feature *= lifter_coeffs.astype(dtype=dtype)

    if use_energy:
        feature[:, 0] = signal_log_energy

    if htk_compat:
        energy = feature[:, 0].unsqueeze(1)  # (m, 1)
        feature = feature[:, 1:]  # (m, n_mfcc - 1)
        if not use_energy:
            energy *= math.sqrt(2)

        feature = paddle.concat((feature, energy), axis=1)

    feature = _subtract_column_mean(feature, subtract_mean)
    return feature
