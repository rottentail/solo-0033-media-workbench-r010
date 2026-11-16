"""Bounded stereo-linked spectral restoration with normalized overlap-add."""
from array import array
import cmath
import math
from . import audio_signal

SIZE = 512
HOP = 128
WINDOW = tuple(math.sin(math.pi*(i+.5)/SIZE) for i in range(SIZE))


def fft(values, inverse=False):
    output = [complex(value) for value in values]
    length = len(output)
    if length < 2 or length & (length-1):
        raise ValueError('FFT length must be a power of two')
    target = 0
    for index in range(1, length):
        bit = length >> 1
        while target & bit:
            target ^= bit
            bit >>= 1
        target ^= bit
        if index < target:
            output[index], output[target] = output[target], output[index]
    width = 2
    while width <= length:
        root = cmath.exp((2j if inverse else -2j)*math.pi/width)
        for origin in range(0, length, width):
            factor = 1+0j
            for local in range(width//2):
                left = output[origin+local]
                right = factor*output[origin+local+width//2]
                output[origin+local] = left+right
                output[origin+local+width//2] = left-right
                factor *= root
        width *= 2
    return [value/length for value in output] if inverse else output


def spectrum(pcm):
    if len(pcm) != SIZE*2:
        raise ValueError('spectral analysis needs one complete stereo window')
    return [fft([pcm[2*i+channel]*WINDOW[i] for i in range(SIZE)]) for channel in range(2)]


def windows(handle, frames):
    """Centre each hop in programme coordinates, including both edge paddings."""
    buffer = array('f', [0])*(2*(SIZE-HOP))
    for origin in range(-(SIZE-HOP), frames, HOP):
        incoming = audio_signal.read_frames(handle, min(HOP, max(0, frames-origin-(SIZE-HOP))))
        incoming.extend([0.0]*(2*HOP-len(incoming)))
        buffer.extend(incoming)
        yield origin, spectrum(buffer)
        del buffer[:2*HOP]


def noise_power(handle, frames):
    # Only full windows estimate stationary power; padding would bias the floor.
    sums = [[0.0]*SIZE for _ in range(2)]
    count = 0
    buffer = audio_signal.read_frames(handle, SIZE)
    while len(buffer) == SIZE*2:
        for channel, bins in enumerate(spectrum(buffer)):
            for index, value in enumerate(bins):
                sums[channel][index] += abs(value)**2
        count += 1
        del buffer[:2*HOP]
        buffer.extend(audio_signal.read_frames(handle, HOP))
    if not count:
        raise ValueError('noise selection is shorter than an analysis window')
    return [[value/count for value in channel] for channel in sums]


def _local_windows(handle, frames):
    pending = audio_signal.read_frames(handle, SIZE)
    pending.extend([0.0]*(SIZE*2-len(pending)))
    for origin in range(0, frames, HOP):
        yield origin, spectrum(pending)
        del pending[:HOP*2]
        incoming = audio_signal.read_frames(handle, HOP)
        incoming.extend([0.0]*(HOP*2-len(incoming)))
        pending.extend(incoming)


def _channel_masks(bins, noise, previous, strength, floor, release):
    masks = []
    for channel in range(2):
        updated = []
        for index in range(SIZE):
            amplitude = abs(bins[channel][index])
            reference = noise[channel][index]*SIZE
            desired = max(floor, 1-strength*reference/max(amplitude, 1e-30))
            old = previous[channel][index]
            updated.append(old*release+desired*(1-release))
        masks.append(updated)
    return masks


def blocks(handle, frames, noise, strength, floor, release, block_frames=4096):
    if type(block_frames) is not int or not 1 <= block_frames <= 65536:
        raise ValueError('invalid denoise output block size')
    previous = [[1.0]*SIZE for _ in range(2)]
    overlap = [[0.0]*SIZE for _ in range(2)]
    pending = array('f')
    normalizer = sum(value*value for value in WINDOW)/HOP
    for origin, bins in _local_windows(handle, frames):
        previous = _channel_masks(bins, noise, previous, strength, floor, release)
        restored = []
        for channel in range(2):
            transformed = [value*previous[channel][i] for i, value in enumerate(bins[channel])]
            restored.append(fft(transformed, True))
        for index in range(SIZE):
            for channel in range(2):
                overlap[channel][index] += restored[channel][index].real*WINDOW[index]
        for local in range(HOP):
            if 0 <= origin+local < frames:
                for channel in range(2):
                    pending.append(overlap[channel][local]/normalizer)
        for channel in range(2):
            overlap[channel] = overlap[channel][HOP:]+[0.0]*HOP
        while len(pending) >= block_frames*2:
            yield pending[:block_frames*2]
            del pending[:block_frames*2]
    if pending:
        yield pending
