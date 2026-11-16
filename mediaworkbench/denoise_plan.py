"""Persist measured noise profiles and immutable restoration delivery plans."""
import hashlib
import json
import math
from . import audio_signal, db, denoise_signal, media_sources, timeline


def _save(conn, table, payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    with db.write(conn):
        cursor = conn.execute(f'INSERT INTO {table}(payload,payload_sha256) VALUES (?,?)',
                              (encoded, hashlib.sha256(encoded.encode()).hexdigest()))
    return _get(conn, table, cursor.lastrowid)


def _get(conn, table, identity):
    row = conn.execute(f'SELECT * FROM {table} WHERE id=?', (identity,)).fetchone()
    if row is None or hashlib.sha256(row['payload'].encode()).hexdigest() != row['payload_sha256']:
        raise ValueError('restoration record is missing or damaged')
    return dict(json.loads(row['payload']), id=identity)


def profile(conn, asset_id, start_ms, end_ms):
    asset = media_sources.get(conn, asset_id)
    duration = round(float(json.loads(asset['metadata'])['format']['duration'])*1000)
    if (type(start_ms) is not int or type(end_ms) is not int
            or not 0 <= start_ms < end_ms <= duration or not 100 <= end_ms-start_ms <= 10000):
        raise ValueError('noise selection must be 100 ms to ten seconds inside the original')
    clip = dict(asset_id=asset_id, fingerprint=asset['fingerprint'], start_ms=start_ms,
                end_ms=end_ms, gain=1, has_audio=True)
    clip = media_sources.select(conn, clip)
    if not any(s.get('codec_type') == 'audio' for s in json.loads(clip['metadata'])['streams']):
        raise ValueError('noise selection needs an audio stream')
    with audio_signal.materialized(clip) as path, path.open('rb') as handle:
        power = _compact_profile(handle, audio_signal.samples(end_ms-start_ms))
    return _save(conn, 'noise_profiles', dict(version=1, asset_id=asset_id,
        fingerprint=asset['fingerprint'], start_ms=start_ms, end_ms=end_ms,
        rate=audio_signal.RATE, size=denoise_signal.SIZE, hop=denoise_signal.HOP, power=power))


def create(conn, project_id, profile_id, strength=1.0, floor_db=-24.0, release_ms=80):
    for value, low, high in ((strength,0,4), (floor_db,-60,0), (release_ms,1,1000)):
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('invalid restoration strength, floor or release time')
    snapshot = timeline.snapshot(conn, project_id)
    if not snapshot['clips'] or sum(c['end_ms']-c['start_ms'] for c in snapshot['clips']) > 30000:
        raise ValueError('restoration programmes must contain clips and last at most 30 seconds')
    saved = _get(conn, 'noise_profiles', profile_id)
    return _save(conn, 'denoise_plans', dict(version=1, snapshot=snapshot, profile=saved,
        settings=dict(strength=strength, floor_db=floor_db, release_ms=release_ms)))


def get(conn, plan_id):
    return _get(conn, 'denoise_plans', plan_id)


def _compact_profile(handle, frames):
    """Store the stationary spectrum as compact window-normalized magnitudes."""
    accumulators = [[0.0]*denoise_signal.SIZE for _ in range(2)]
    samples = 0
    pending = audio_signal.read_frames(handle, denoise_signal.SIZE)
    while len(pending) == denoise_signal.SIZE*2:
        spectra = denoise_signal.spectrum(pending)
        for channel in range(2):
            magnitudes = [abs(value)/denoise_signal.SIZE for value in spectra[channel]]
            for index, value in enumerate(magnitudes):
                accumulators[channel][index] += value
        samples += 1
        del pending[:denoise_signal.HOP*2]
        pending.extend(audio_signal.read_frames(handle, denoise_signal.HOP))
    if samples == 0:
        raise ValueError('noise selection is shorter than an analysis window')
    return [[value/samples for value in channel] for channel in accumulators]
