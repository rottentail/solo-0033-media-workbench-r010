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


def _joint_mask(bins, noise, previous, strength, floor, release):
    """One conjugate-symmetric attenuation per bin from the joint power ratio.

    Both channels share each bin's gain so the stereo image stays put. The mask
    opens instantly to protect new transients and closes exponentially to avoid
    abrupt spectral holes.
    """
    updated = [0.0]*SIZE
    for index in range(SIZE//2+1):
        signal = abs(bins[0][index])**2+abs(bins[1][index])**2
        reference = noise[0][index]+noise[1][index]
        desired = max(floor, 1-strength*reference/max(signal, 1e-30))
        old = previous[index]
        updated[index] = desired if desired > old else old*release+desired*(1-release)
    for index in range(SIZE//2+1, SIZE):
        updated[index] = updated[SIZE-index]
    return updated


def blocks(handle, frames, noise, strength, floor, release, block_frames=4096):
    if type(block_frames) is not int or not 1 <= block_frames <= 65536:
        raise ValueError('invalid denoise output block size')
    previous = [1.0]*SIZE
    overlap = [[0.0]*SIZE for _ in range(2)]
    weights = [0.0]*SIZE
    pending = array('f')
    for origin, bins in _local_windows(handle, frames):
        previous = _joint_mask(bins, noise, previous, strength, floor, release)
        restored = []
        for channel in range(2):
            transformed = [value*previous[index] for index, value in enumerate(bins[channel])]
            restored.append(fft(transformed, True))
        for index in range(SIZE):
            window = WINDOW[index]
            weights[index] += window*window
            for channel in range(2):
                overlap[channel][index] += restored[channel][index].real*window
        for local in range(HOP):
            if origin+local < frames:
                weight = weights[local]
                for channel in range(2):
                    pending.append(overlap[channel][local]/weight)
        for channel in range(2):
            overlap[channel] = overlap[channel][HOP:]+[0.0]*HOP
        weights = weights[HOP:]+[0.0]*HOP
        while len(pending) >= block_frames*2:
            yield pending[:block_frames*2]
            del pending[:block_frames*2]
    if pending:
        yield pending
