from collections import OrderedDict

import librosa
import tqdm
import json
import pathlib
import torch.nn.functional as F
import numpy as np
import torch
from typing import Dict
from librosa.filters import mel as librosa_mel_fn
from basics.base_svs_infer import BaseSVSInfer
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.fastspeech.tts_modules import LengthRegulator
from modules.toplevel import DiffSingerAcoustic, ShallowDiffusionOutput
from modules.vocoders.registry import VOCODERS
from utils import load_ckpt
from utils.hparams import hparams
from utils.infer_utils import cross_fade, resample_align_curve, save_wav
from utils.phoneme_utils import build_phoneme_list
from utils.text_encoder import TokenTextEncoder
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
class STFT:
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025,
                 clip_val=1e-5):
        self.target_sr = sr

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False):
        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val

        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))

        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        mel_basis_key = str(fmax) + '_' + str(y.device)
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        keyshift_key = str(keyshift) + '_' + str(y.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)

        y = torch.nn.functional.pad(y.unsqueeze(1),
                                    ((win_size_new - hop_length_new) // 2, (win_size_new - hop_length_new + 1) // 2),
                                    mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(
            y, n_fft_new, hop_length=hop_length_new,
            win_length=win_size_new, window=self.hann_window[keyshift_key],
            center=center, pad_mode='reflect',
            normalized=False, onesided=True, return_complex=True
        ).abs()
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * win_size / win_size_new

        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)

        return spec

    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect
def load_wav_to_torch(full_path, target_sr=None):
    data, sr = librosa.load(full_path, sr=target_sr, mono=True)
    return torch.from_numpy(data), sr

def wav2spec(inp_path, keyshift=0, speed=1, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampling_rate = hparams['audio_sample_rate']
    num_mels = hparams['audio_num_mel_bins']
    n_fft = hparams['fft_size']
    win_size = hparams['win_size']
    hop_size = hparams['hop_size']
    fmin = hparams['fmin']
    fmax = hparams['fmax']
    stft = STFT(sampling_rate, num_mels, n_fft, win_size, hop_size, fmin, fmax)
    with torch.no_grad():
        wav_torch, _ = load_wav_to_torch(inp_path, target_sr=stft.target_sr)
        mel_torch = stft.get_mel(wav_torch.unsqueeze(0).to(device), keyshift=keyshift, speed=speed).squeeze(0).T
        # log mel to log10 mel
        mel_torch = 0.434294 * mel_torch
        return wav_torch.cpu().numpy(), mel_torch.cpu().numpy()



class DiffSingerAcousticInfer(BaseSVSInfer):
    def __init__(self, device=None, load_model=True, load_vocoder=True, ckpt_steps=None):
        super().__init__(device=device)
        if load_model:
            self.variance_checklist = []

            self.variances_to_embed = set()

            if hparams.get('use_energy_embed', False):
                self.variances_to_embed.add('energy')
            if hparams.get('use_breathiness_embed', False):
                self.variances_to_embed.add('breathiness')

            self.ph_encoder = TokenTextEncoder(vocab_list=build_phoneme_list())
            if hparams['use_spk_id']:
                with open(pathlib.Path(hparams['work_dir']) / 'spk_map.json', 'r', encoding='utf8') as f:
                    self.spk_map = json.load(f)
                assert isinstance(self.spk_map, dict) and len(self.spk_map) > 0, 'Invalid or empty speaker map!'
                assert len(self.spk_map) == len(set(self.spk_map.values())), 'Duplicate speaker id in speaker map!'
            self.model = self.build_model(ckpt_steps=ckpt_steps)
            self.lr = LengthRegulator().to(self.device)
        if load_vocoder:
            self.vocoder = self.build_vocoder()

    def build_model(self, ckpt_steps=None):
        model = DiffSingerAcoustic(
            vocab_size=len(self.ph_encoder),
            out_dims=hparams['audio_num_mel_bins']
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], ckpt_steps=ckpt_steps,
                  prefix_in_ckpt='model', strict=True, device=self.device)
        return model

    def build_vocoder(self):
        if hparams['vocoder'] in VOCODERS:
            vocoder = VOCODERS[hparams['vocoder']]()
        else:
            vocoder = VOCODERS[hparams['vocoder'].split('.')[-1]]()
        vocoder.to_device(self.device)
        return vocoder

    def preprocess_input(self, param, idx=0):
        """
        :param param: one segment in the .ds file
        :param idx: index of the segment
        :return: batch of the model inputs
        """
        batch = {}
        summary = OrderedDict()
        txt_tokens = torch.LongTensor([self.ph_encoder.encode(param['ph_seq'])]).to(self.device)  # => [B, T_txt]
        batch['tokens'] = txt_tokens

        ph_dur = torch.from_numpy(np.array(param['ph_dur'].split(), np.float32)).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur, dim=0) / self.timestep + 0.5).long()
        durations = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))[None]  # => [B=1, T_txt]
        mel2ph = self.lr(durations, txt_tokens == 0)  # => [B=1, T]
        batch['mel2ph'] = mel2ph
        length = mel2ph.size(1)  # => T

        summary['tokens'] = txt_tokens.size(1)
        summary['frames'] = length
        summary['seconds'] = '%.2f' % (length * self.timestep)

        if hparams['use_spk_id']:
            spk_mix_id, spk_mix_value = self.load_speaker_mix(
                param_src=param, summary_dst=summary, mix_mode='frame', mix_length=length
            )
            batch['spk_mix_id'] = spk_mix_id
            batch['spk_mix_value'] = spk_mix_value

        batch['f0'] = torch.from_numpy(resample_align_curve(
            np.array(param['f0_seq'].split(), np.float32),
            original_timestep=float(param['f0_timestep']),
            target_timestep=self.timestep,
            align_length=length
        )).to(self.device)[None]

        for v_name in VARIANCE_CHECKLIST:
            if v_name in self.variances_to_embed:
                batch[v_name] = torch.from_numpy(resample_align_curve(
                    np.array(param[v_name].split(), np.float32),
                    original_timestep=float(param[f'{v_name}_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )).to(self.device)[None]
                summary[v_name] = 'manual'

        if hparams.get('use_key_shift_embed', False):
            shift_min, shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
            gender = param.get('gender')
            if gender is None:
                gender = 0.
            if isinstance(gender, (int, float, bool)):  # static gender value
                summary['gender'] = f'static({gender:.3f})'
                key_shift_value = gender * shift_max if gender >= 0 else gender * abs(shift_min)
                batch['key_shift'] = torch.FloatTensor([key_shift_value]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                summary['gender'] = 'dynamic'
                gender_seq = resample_align_curve(
                    np.array(gender.split(), np.float32),
                    original_timestep=float(param['gender_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )
                gender_mask = gender_seq >= 0
                key_shift_seq = gender_seq * (gender_mask * shift_max + (1 - gender_mask) * abs(shift_min))
                batch['key_shift'] = torch.clip(
                    torch.from_numpy(key_shift_seq.astype(np.float32)).to(self.device)[None],  # => [B=1, T]
                    min=shift_min, max=shift_max
                )

        if hparams.get('use_speed_embed', False):
            if param.get('velocity') is None:
                summary['velocity'] = 'default'
                batch['speed'] = torch.FloatTensor([1.]).to(self.device)[:, None]  # => [B=1, T=1]
            else:
                summary['velocity'] = 'manual'
                speed_min, speed_max = hparams['augmentation_args']['random_time_stretching']['range']
                speed_seq = resample_align_curve(
                    np.array(param['velocity'].split(), np.float32),
                    original_timestep=float(param['velocity_timestep']),
                    target_timestep=self.timestep,
                    align_length=length
                )
                batch['speed'] = torch.clip(
                    torch.from_numpy(speed_seq.astype(np.float32)).to(self.device)[None],  # => [B=1, T]
                    min=speed_min, max=speed_max
                )

        print(f'[{idx}]\t' + ', '.join(f'{k}: {v}' for k, v in summary.items()))

        return batch

    @torch.no_grad()
    def forward_model(self, sample,ppm):
        txt_tokens = sample['tokens']
        variances = {
            v_name: sample.get(v_name)
            for v_name in self.variances_to_embed
        }
        if hparams['use_spk_id']:
            spk_mix_id = sample['spk_mix_id']
            spk_mix_value = sample['spk_mix_value']
            # perform mixing on spk embed
            spk_mix_embed = torch.sum(
                self.model.fs2.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(3),  # => [B, T, N, H]
                dim=2, keepdim=False
            )  # => [B, T, H]
        else:
            spk_mix_embed = None
        mel_pred: ShallowDiffusionOutput = self.model(
            txt_tokens, mel2ph=sample['mel2ph'], f0=sample['f0'], **variances,
            key_shift=sample.get('key_shift'), speed=sample.get('speed'),
            spk_mix_embed=spk_mix_embed, infer=True,promot=ppm
        )
        return mel_pred.diff_out

    @torch.no_grad()
    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

    def run_inference(
            self, params,
            out_dir: pathlib.Path = None,
            title: str = None,
            num_runs: int = 1,
            spk_mix: Dict[str, float] = None,
            seed: int = -1,
            save_mel: bool = False,ppm:str=None,maxppm=None


    ):

        wva,melss=wav2spec(ppm,device='cpu')
        melss=torch.from_numpy(melss).cuda()
        if maxppm is not None:
            melss=melss[:maxppm,:]

        # melss:torch.tensor()
        melss=torch.unsqueeze(melss, dim=0)
        batches = [self.preprocess_input(param, idx=i) for i, param in enumerate(params)]

        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = '.wav' if not save_mel else '.mel.pt'
        for i in range(num_runs):
            if save_mel:
                result = []
            else:
                result = np.zeros(0)
            current_length = 0

            for param, batch in tqdm.tqdm(
                    zip(params, batches), desc='infer segments', total=len(params)
            ):
                if 'seed' in param:
                    torch.manual_seed(param["seed"] & 0xffff_ffff)
                    torch.cuda.manual_seed_all(param["seed"] & 0xffff_ffff)
                elif seed >= 0:
                    torch.manual_seed(seed & 0xffff_ffff)
                    torch.cuda.manual_seed_all(seed & 0xffff_ffff)

                mel_pred = self.forward_model(batch,melss)
                if save_mel:
                    result.append({
                        'offset': param.get('offset', 0.),
                        'mel': mel_pred.cpu(),
                        'f0': batch['f0'].cpu()
                    })
                else:
                    waveform_pred = self.run_vocoder(mel_pred, f0=batch['f0'])[0].cpu().numpy()
                    silent_length = round(param.get('offset', 0) * hparams['audio_sample_rate']) - current_length
                    if silent_length >= 0:
                        result = np.append(result, np.zeros(silent_length))
                        result = np.append(result, waveform_pred)
                    else:
                        result = cross_fade(result, waveform_pred, current_length + silent_length)
                    current_length = current_length + silent_length + waveform_pred.shape[0]

            if num_runs > 1:
                filename = f'{title}-{str(i).zfill(3)}{suffix}'
            else:
                filename = title + suffix
            save_path = out_dir / filename
            if save_mel:
                print(f'| save mel: {save_path}')
                torch.save(result, save_path)
            else:
                print(f'| save audio: {save_path}')
                save_wav(result, save_path, hparams['audio_sample_rate'])
