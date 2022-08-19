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

import os
import warnings
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import resampy
import soundfile
from scipy.io import wavfile

from ..utils import depth_convert
from ..utils import ParameterError
from .common import AudioMetaData

__all__ = [
    'resample',
    'to_mono',
    'normalize',
    'save',
    'soundfile_save',
    'load',
    'soundfile_load',
    'info',
    'to_mono'
]
NORMALMIZE_TYPES = ['linear', 'gaussian']
MERGE_TYPES = ['ch0', 'ch1', 'random', 'average']
RESAMPLE_MODES = ['kaiser_best', 'kaiser_fast']
EPS = 1e-8


def resample(y: np.ndarray,
             src_sr: int,
             target_sr: int,
             mode: str='kaiser_fast') -> np.ndarray:
    """Audio resampling.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        src_sr (int): Source sample rate.
        target_sr (int): Target sample rate.
        mode (str, optional): The resampling filter to use. Defaults to 'kaiser_fast'.

    Returns:
        np.ndarray: `y` resampled to `target_sr`
    """

    if mode == 'kaiser_best':
        warnings.warn(
            f'Using resampy in kaiser_best to {src_sr}=>{target_sr}. This function is pretty slow, \
        we recommend the mode kaiser_fast in large scale audio trainning')

    if not isinstance(y, np.ndarray):
        raise ParameterError(
            'Only support numpy np.ndarray, but received y in {type(y)}')

    if mode not in RESAMPLE_MODES:
        raise ParameterError(f'resample mode must in {RESAMPLE_MODES}')

    return resampy.resample(y, src_sr, target_sr, filter=mode)


def to_mono(y: np.ndarray, merge_type: str='average') -> np.ndarray:
    """Convert sterior audio to mono.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        merge_type (str, optional): Merge type to generate mono waveform. Defaults to 'average'.

    Returns:
        np.ndarray: `y` with mono channel.
    """

    if merge_type not in MERGE_TYPES:
        raise ParameterError(
            f'Unsupported merge type {merge_type}, available types are {MERGE_TYPES}'
        )
    if y.ndim > 2:
        raise ParameterError(
            f'Unsupported audio array,  y.ndim > 2, the shape is {y.shape}')
    if y.ndim == 1:  # nothing to merge
        return y

    if merge_type == 'ch0':
        return y[0]
    if merge_type == 'ch1':
        return y[1]
    if merge_type == 'random':
        return y[np.random.randint(0, 2)]

    # need to do averaging according to dtype

    if y.dtype == 'float32':
        y_out = (y[0] + y[1]) * 0.5
    elif y.dtype == 'int16':
        y_out = y.astype('int32')
        y_out = (y_out[0] + y_out[1]) // 2
        y_out = np.clip(y_out, np.iinfo(y.dtype).min,
                        np.iinfo(y.dtype).max).astype(y.dtype)

    elif y.dtype == 'int8':
        y_out = y.astype('int16')
        y_out = (y_out[0] + y_out[1]) // 2
        y_out = np.clip(y_out, np.iinfo(y.dtype).min,
                        np.iinfo(y.dtype).max).astype(y.dtype)
    else:
        raise ParameterError(f'Unsupported dtype: {y.dtype}')
    return y_out


def soundfile_load_(file: os.PathLike,
                    offset: Optional[float]=None,
                    dtype: str='int16',
                    duration: Optional[int]=None) -> Tuple[np.ndarray, int]:
    """Load audio using soundfile library. This function load audio file using libsndfile.

    Args:
        file (os.PathLike): File of waveform.
        offset (Optional[float], optional): Offset to the start of waveform. Defaults to None.
        dtype (str, optional): Data type of waveform. Defaults to 'int16'.
        duration (Optional[int], optional): Duration of waveform to read. Defaults to None.

    Returns:
        Tuple[np.ndarray, int]: Waveform in ndarray and its samplerate.
    """
    with soundfile.SoundFile(file) as sf_desc:
        sr_native = sf_desc.samplerate
        if offset:
            sf_desc.seek(int(offset * sr_native))
        if duration is not None:
            frame_duration = int(duration * sr_native)
        else:
            frame_duration = -1
        y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T

    return y, sf_desc.samplerate


def normalize(y: np.ndarray, norm_type: str='linear',
              mul_factor: float=1.0) -> np.ndarray:
    """Normalize an input audio with additional multiplier.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        norm_type (str, optional): Type of normalization. Defaults to 'linear'.
        mul_factor (float, optional): Scaling factor. Defaults to 1.0.

    Returns:
        np.ndarray: `y` after normalization.
    """

    if norm_type == 'linear':
        amax = np.max(np.abs(y))
        factor = 1.0 / (amax + EPS)
        y = y * factor * mul_factor
    elif norm_type == 'gaussian':
        amean = np.mean(y)
        astd = np.std(y)
        astd = max(astd, EPS)
        y = mul_factor * (y - amean) / astd
    else:
        raise NotImplementedError(f'norm_type should be in {NORMALMIZE_TYPES}')

    return y


def soundfile_save(y: np.ndarray, sr: int, file: os.PathLike) -> None:
    """Save audio file to disk. This function saves audio to disk using scipy.io.wavfile, with additional step to convert input waveform to int16.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        sr (int): Sample rate.
        file (os.PathLike): Path of auido file to save.
    """
    if not file.endswith('.wav'):
        raise ParameterError(
            f'only .wav file supported, but dst file name is: {file}')

    if sr <= 0:
        raise ParameterError(
            f'Sample rate should be larger than 0, recieved sr = {sr}')

    if y.dtype not in ['int16', 'int8']:
        warnings.warn(
            f'input data type is {y.dtype}, will convert data to int16 format before saving'
        )
        y_out = depth_convert(y, 'int16')
    else:
        y_out = y

    wavfile.write(file, sr, y_out)

def soundfile_load(
        file: os.PathLike,
        sr: Optional[int]=None,
        mono: bool=True,
        merge_type: str='average',  # ch0,ch1,random,average
        normal: bool=True,
        norm_type: str='linear',
        norm_mul_factor: float=1.0,
        offset: float=0.0,
        duration: Optional[int]=None,
        dtype: str='float32',
        resample_mode: str='kaiser_fast') -> Tuple[np.ndarray, int]:
    """Load audio file from disk. This function loads audio from disk using using audio beackend.

    Args:
        file (os.PathLike): Path of auido file to load.
        sr (Optional[int], optional): Sample rate of loaded waveform. Defaults to None.
        mono (bool, optional): Return waveform with mono channel. Defaults to True.
        merge_type (str, optional): Merge type of multi-channels waveform. Defaults to 'average'.
        normal (bool, optional): Waveform normalization. Defaults to True.
        norm_type (str, optional): Type of normalization. Defaults to 'linear'.
        norm_mul_factor (float, optional): Scaling factor. Defaults to 1.0.
        offset (float, optional): Offset to the start of waveform. Defaults to 0.0.
        duration (Optional[int], optional): Duration of waveform to read. Defaults to None.
        dtype (str, optional): Data type of waveform. Defaults to 'float32'.
        resample_mode (str, optional): The resampling filter to use. Defaults to 'kaiser_fast'.

    Returns:
        Tuple[np.ndarray, int]: Waveform in ndarray and its samplerate.
    """

    y, r = soundfile_load_(file, offset=offset, dtype=dtype, duration=duration)

    if not ((y.ndim == 1 and len(y) > 0) or (y.ndim == 2 and len(y[0]) > 0)):
        raise ParameterError(f'audio file {file} looks empty')

    if mono:
        y = to_mono(y, merge_type)

    if sr is not None and sr != r:
        y = resample(y, r, sr, mode=resample_mode)
        r = sr

    if normal:
        y = normalize(y, norm_type, norm_mul_factor)
    elif dtype in ['int8', 'int16']:
        # still need to do normalization, before depth convertion
        y = normalize(y, 'linear', 1.0)

    y = depth_convert(y, dtype)
    return y, r

#the code below is form: https://github.com/pytorch/audio/blob/main/torchaudio/backend/soundfile_backend.py

def _get_subtype_for_wav(dtype: paddle.dtype, encoding: str, bits_per_sample: int):
    if not encoding:
        if not bits_per_sample:
            subtype = {
                paddle.uint8: "PCM_U8",
                paddle.int16: "PCM_16",
                paddle.int32: "PCM_32",
                paddle.float32: "FLOAT",
                paddle.float64: "DOUBLE",
            }.get(dtype)
            if not subtype:
                raise ValueError(f"Unsupported dtype for wav: {dtype}")
            return subtype
        if bits_per_sample == 8:
            return "PCM_U8"
        return f"PCM_{bits_per_sample}"
    if encoding == "PCM_S":
        if not bits_per_sample:
            return "PCM_32"
        if bits_per_sample == 8:
            raise ValueError("wav does not support 8-bit signed PCM encoding.")
        return f"PCM_{bits_per_sample}"
    if encoding == "PCM_U":
        if bits_per_sample in (None, 8):
            return "PCM_U8"
        raise ValueError("wav only supports 8-bit unsigned PCM encoding.")
    if encoding == "PCM_F":
        if bits_per_sample in (None, 32):
            return "FLOAT"
        if bits_per_sample == 64:
            return "DOUBLE"
        raise ValueError("wav only supports 32/64-bit float PCM encoding.")
    if encoding == "ULAW":
        if bits_per_sample in (None, 8):
            return "ULAW"
        raise ValueError("wav only supports 8-bit mu-law encoding.")
    if encoding == "ALAW":
        if bits_per_sample in (None, 8):
            return "ALAW"
        raise ValueError("wav only supports 8-bit a-law encoding.")
    raise ValueError(f"wav does not support {encoding}.")


def _get_subtype_for_sphere(encoding: str, bits_per_sample: int):
    if encoding in (None, "PCM_S"):
        return f"PCM_{bits_per_sample}" if bits_per_sample else "PCM_32"
    if encoding in ("PCM_U", "PCM_F"):
        raise ValueError(f"sph does not support {encoding} encoding.")
    if encoding == "ULAW":
        if bits_per_sample in (None, 8):
            return "ULAW"
        raise ValueError("sph only supports 8-bit for mu-law encoding.")
    if encoding == "ALAW":
        return "ALAW"
    raise ValueError(f"sph does not support {encoding}.")


def _get_subtype(dtype: paddle.dtype, format: str, encoding: str, bits_per_sample: int):
    if format == "wav":
        return _get_subtype_for_wav(dtype, encoding, bits_per_sample)
    if format == "flac":
        if encoding:
            raise ValueError("flac does not support encoding.")
        if not bits_per_sample:
            return "PCM_16"
        if bits_per_sample > 24:
            raise ValueError("flac does not support bits_per_sample > 24.")
        return "PCM_S8" if bits_per_sample == 8 else f"PCM_{bits_per_sample}"
    if format in ("ogg", "vorbis"):
        if encoding or bits_per_sample:
            raise ValueError("ogg/vorbis does not support encoding/bits_per_sample.")
        return "VORBIS"
    if format == "sph":
        return _get_subtype_for_sphere(encoding, bits_per_sample)
    if format in ("nis", "nist"):
        return "PCM_16"
    raise ValueError(f"Unsupported format: {format}")

def save(
    filepath: str,
    src: paddle.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    compression: Optional[float] = None,
    format: Optional[str] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
):
    """Save audio data to file.

    Note:
        The formats this function can handle depend on the soundfile installation.
        This function is tested on the following formats;

        * WAV

            * 32-bit floating-point
            * 32-bit signed integer
            * 16-bit signed integer
            * 8-bit unsigned integer

        * FLAC
        * OGG/VORBIS
        * SPHERE

    Note:
        ``filepath`` argument is intentionally annotated as ``str`` only, even though it accepts
        ``pathlib.Path`` object as well. This is for the consistency with ``"sox_io"`` backend,

    Args:
        filepath (str or pathlib.Path): Path to audio file.
        src (paddle.Tensor): Audio data to save. must be 2D tensor.
        sample_rate (int): sampling rate
        channels_first (bool, optional): If ``True``, the given tensor is interpreted as `[channel, time]`,
            otherwise `[time, channel]`.
        compression (float of None, optional): Not used.
            It is here only for interface compatibility reson with "sox_io" backend.
        format (str or None, optional): Override the audio format.
            When ``filepath`` argument is path-like object, audio format is
            inferred from file extension. If the file extension is missing or
            different, you can specify the correct format with this argument.

            When ``filepath`` argument is file-like object,
            this argument is required.

            Valid values are ``"wav"``, ``"ogg"``, ``"vorbis"``,
            ``"flac"`` and ``"sph"``.
        encoding (str or None, optional): Changes the encoding for supported formats.
            This argument is effective only for supported formats, sush as
            ``"wav"``, ``""flac"`` and ``"sph"``. Valid values are;

                - ``"PCM_S"`` (signed integer Linear PCM)
                - ``"PCM_U"`` (unsigned integer Linear PCM)
                - ``"PCM_F"`` (floating point PCM)
                - ``"ULAW"`` (mu-law)
                - ``"ALAW"`` (a-law)

        bits_per_sample (int or None, optional): Changes the bit depth for the
            supported formats.
            When ``format`` is one of ``"wav"``, ``"flac"`` or ``"sph"``,
            you can change the bit depth.
            Valid values are ``8``, ``16``, ``24``, ``32`` and ``64``.

    Supported formats/encodings/bit depth/compression are:

    ``"wav"``
        - 32-bit floating-point PCM
        - 32-bit signed integer PCM
        - 24-bit signed integer PCM
        - 16-bit signed integer PCM
        - 8-bit unsigned integer PCM
        - 8-bit mu-law
        - 8-bit a-law

        Note:
            Default encoding/bit depth is determined by the dtype of
            the input Tensor.

    ``"flac"``
        - 8-bit
        - 16-bit (default)
        - 24-bit

    ``"ogg"``, ``"vorbis"``
        - Doesn't accept changing configuration.

    ``"sph"``
        - 8-bit signed integer PCM
        - 16-bit signed integer PCM
        - 24-bit signed integer PCM
        - 32-bit signed integer PCM (default)
        - 8-bit mu-law
        - 8-bit a-law
        - 16-bit a-law
        - 24-bit a-law
        - 32-bit a-law

    """
    if src.ndim != 2:
        raise ValueError(f"Expected 2D Tensor, got {src.ndim}D.")
    if compression is not None:
        warnings.warn(
            '`save` function of "soundfile" backend does not support "compression" parameter. '
            "The argument is silently ignored."
        )
    if hasattr(filepath, "write"):
        if format is None:
            raise RuntimeError("`format` is required when saving to file object.")
        ext = format.lower()
    else:
        ext = str(filepath).split(".")[-1].lower()

    if bits_per_sample not in (None, 8, 16, 24, 32, 64):
        raise ValueError("Invalid bits_per_sample.")
    if bits_per_sample == 24:
        warnings.warn(
            "Saving audio with 24 bits per sample might warp samples near -1. "
            "Using 16 bits per sample might be able to avoid this."
        )
    subtype = _get_subtype(src.dtype, ext, encoding, bits_per_sample)

    # sph is a extension used in TED-LIUM but soundfile does not recognize it as NIST format,
    # so we extend the extensions manually here
    if ext in ["nis", "nist", "sph"] and format is None:
        format = "NIST"

    if channels_first:
        src = src.t()

    soundfile.write(file=filepath, data=src, samplerate=sample_rate, subtype=subtype, format=format)

_SUBTYPE2DTYPE = {
    "PCM_S8": "int8",
    "PCM_U8": "uint8",
    "PCM_16": "int16",
    "PCM_32": "int32",
    "FLOAT": "float32",
    "DOUBLE": "float64",
}

def load(
    filepath: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
) -> Tuple[paddle.Tensor, int]:
    """Load audio data from file.

    Note:
        The formats this function can handle depend on the soundfile installation.
        This function is tested on the following formats;

        * WAV

            * 32-bit floating-point
            * 32-bit signed integer
            * 16-bit signed integer
            * 8-bit unsigned integer

        * FLAC
        * OGG/VORBIS
        * SPHERE

    By default (``normalize=True``, ``channels_first=True``), this function returns Tensor with
    ``float32`` dtype and the shape of `[channel, time]`.
    The samples are normalized to fit in the range of ``[-1.0, 1.0]``.

    When the input format is WAV with integer type, such as 32-bit signed integer, 16-bit
    signed integer and 8-bit unsigned integer (24-bit signed integer is not supported),
    by providing ``normalize=False``, this function can return integer Tensor, where the samples
    are expressed within the whole range of the corresponding dtype, that is, ``int32`` tensor
    for 32-bit signed PCM, ``int16`` for 16-bit signed PCM and ``uint8`` for 8-bit unsigned PCM.

    ``normalize`` parameter has no effect on 32-bit floating-point WAV and other formats, such as
    ``flac`` and ``mp3``.
    For these formats, this function always returns ``float32`` Tensor with values normalized to
    ``[-1.0, 1.0]``.

    Note:
        ``filepath`` argument is intentionally annotated as ``str`` only, even though it accepts
        ``pathlib.Path`` object as well. This is for the consistency with ``"sox_io"`` backend.

    Args:
        filepath (path-like object or file-like object):
            Source of audio data.
        frame_offset (int, optional):
            Number of frames to skip before start reading data.
        num_frames (int, optional):
            Maximum number of frames to read. ``-1`` reads all the remaining samples,
            starting from ``frame_offset``.
            This function may return the less number of frames if there is not enough
            frames in the given file.
        normalize (bool, optional):
            When ``True``, this function always return ``float32``, and sample values are
            normalized to ``[-1.0, 1.0]``.
            If input file is integer WAV, giving ``False`` will change the resulting Tensor type to
            integer type.
            This argument has no effect for formats other than integer WAV type.
        channels_first (bool, optional):
            When True, the returned Tensor has dimension `[channel, time]`.
            Otherwise, the returned Tensor's dimension is `[time, channel]`.
        format (str or None, optional):
            Not used. PySoundFile does not accept format hint.

    Returns:
        (paddle.Tensor, int): Resulting Tensor and sample rate.
            If the input file has integer wav format and normalization is off, then it has
            integer type, else ``float32`` type. If ``channels_first=True``, it has
            `[channel, time]` else `[time, channel]`.
    """
    with soundfile.SoundFile(filepath, "r") as file_:
        if file_.format != "WAV" or normalize:
            dtype = "float32"
        elif file_.subtype not in _SUBTYPE2DTYPE:
            raise ValueError(f"Unsupported subtype: {file_.subtype}")
        else:
            dtype = _SUBTYPE2DTYPE[file_.subtype]

        frames = file_._prepare_read(frame_offset, None, num_frames)
        waveform = file_.read(frames, dtype, always_2d=True)
        sample_rate = file_.samplerate

    waveform = paddle.to_tensor(waveform)
    if channels_first:
        waveform = paddle.transpose(waveform, perm=[1,0])
    return waveform, sample_rate


# Mapping from soundfile subtype to number of bits per sample.
# This is mostly heuristical and the value is set to 0 when it is irrelevant
# (lossy formats) or when it can't be inferred.
# For ADPCM (and G72X) subtypes, it's hard to infer the bit depth because it's not part of the standard:
# According to https://en.wikipedia.org/wiki/Adaptive_differential_pulse-code_modulation#In_telephony,
# the default seems to be 8 bits but it can be compressed further to 4 bits.
# The dict is inspired from
# https://github.com/bastibe/python-soundfile/blob/744efb4b01abc72498a96b09115b42a4cabd85e4/soundfile.py#L66-L94
_SUBTYPE_TO_BITS_PER_SAMPLE = {
    "PCM_S8": 8,  # Signed 8 bit data
    "PCM_16": 16,  # Signed 16 bit data
    "PCM_24": 24,  # Signed 24 bit data
    "PCM_32": 32,  # Signed 32 bit data
    "PCM_U8": 8,  # Unsigned 8 bit data (WAV and RAW only)
    "FLOAT": 32,  # 32 bit float data
    "DOUBLE": 64,  # 64 bit float data
    "ULAW": 8,  # U-Law encoded. See https://en.wikipedia.org/wiki/G.711#Types
    "ALAW": 8,  # A-Law encoded. See https://en.wikipedia.org/wiki/G.711#Types
    "IMA_ADPCM": 0,  # IMA ADPCM.
    "MS_ADPCM": 0,  # Microsoft ADPCM.
    "GSM610": 0,  # GSM 6.10 encoding. (Wikipedia says 1.625 bit depth?? https://en.wikipedia.org/wiki/Full_Rate)
    "VOX_ADPCM": 0,  # OKI / Dialogix ADPCM
    "G721_32": 0,  # 32kbs G721 ADPCM encoding.
    "G723_24": 0,  # 24kbs G723 ADPCM encoding.
    "G723_40": 0,  # 40kbs G723 ADPCM encoding.
    "DWVW_12": 12,  # 12 bit Delta Width Variable Word encoding.
    "DWVW_16": 16,  # 16 bit Delta Width Variable Word encoding.
    "DWVW_24": 24,  # 24 bit Delta Width Variable Word encoding.
    "DWVW_N": 0,  # N bit Delta Width Variable Word encoding.
    "DPCM_8": 8,  # 8 bit differential PCM (XI only)
    "DPCM_16": 16,  # 16 bit differential PCM (XI only)
    "VORBIS": 0,  # Xiph Vorbis encoding. (lossy)
    "ALAC_16": 16,  # Apple Lossless Audio Codec (16 bit).
    "ALAC_20": 20,  # Apple Lossless Audio Codec (20 bit).
    "ALAC_24": 24,  # Apple Lossless Audio Codec (24 bit).
    "ALAC_32": 32,  # Apple Lossless Audio Codec (32 bit).
}

def _get_bit_depth(subtype):
    if subtype not in _SUBTYPE_TO_BITS_PER_SAMPLE:
        warnings.warn(
            f"The {subtype} subtype is unknown to PaddleAudio. As a result, the bits_per_sample "
            "attribute will be set to 0. If you are seeing this warning, please "
            "report by opening an issue on github (after checking for existing/closed ones). "
            "You may otherwise ignore this warning."
        )
    return _SUBTYPE_TO_BITS_PER_SAMPLE.get(subtype, 0)

_SUBTYPE_TO_ENCODING = {
    "PCM_S8": "PCM_S",
    "PCM_16": "PCM_S",
    "PCM_24": "PCM_S",
    "PCM_32": "PCM_S",
    "PCM_U8": "PCM_U",
    "FLOAT": "PCM_F",
    "DOUBLE": "PCM_F",
    "ULAW": "ULAW",
    "ALAW": "ALAW",
    "VORBIS": "VORBIS",
}

def _get_encoding(format: str, subtype: str):
    if format == "FLAC":
        return "FLAC"
    return _SUBTYPE_TO_ENCODING.get(subtype, "UNKNOWN")

def info(filepath: str, format: Optional[str] = None) -> AudioMetaData:
    """Get signal information of an audio file.

    Note:
        ``filepath`` argument is intentionally annotated as ``str`` only, even though it accepts
        ``pathlib.Path`` object as well. This is for the consistency with ``"sox_io"`` backend,

    Args:
        filepath (path-like object or file-like object):
            Source of audio data.
        format (str or None, optional):
            Not used. PySoundFile does not accept format hint.

    Returns:
        AudioMetaData: meta data of the given audio.

    """
    sinfo = soundfile.info(filepath)
    return AudioMetaData(
        sinfo.samplerate,
        sinfo.frames,
        sinfo.channels,
        bits_per_sample=_get_bit_depth(sinfo.subtype),
        encoding=_get_encoding(sinfo.format, sinfo.subtype),
    )
