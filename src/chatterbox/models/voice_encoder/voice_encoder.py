from typing import List, Optional, Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import librosa
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .config import VoiceEncConfig
from .melspec import melspectrogram


def pack(arrays: list, seq_len: Optional[int] = None, pad_value: float = 0.0) -> Tensor:
    """Packs a list of array-like objects of shape (Ti, ...) into a single tensor of shape (B, T, ...)."""
    max_len = max(len(array) for array in arrays)
    if seq_len is None:
        seq_len = max_len
    else:
        assert seq_len >= max_len

    if isinstance(arrays[0], list):
        arrays = [np.array(array) for array in arrays]

    device = arrays[0].device if isinstance(arrays[0], Tensor) else None
    tensors = [torch.as_tensor(array) for array in arrays]
    
    packed_shape = (len(tensors), seq_len, *tensors[0].shape[1:])
    packed_tensor = torch.full(packed_shape, pad_value, dtype=tensors[0].dtype, device=device)
    
    for i, tensor in enumerate(tensors):
        packed_tensor[i, :tensor.size(0)] = tensor
        
    return packed_tensor


def get_num_wins(n_frames: int, step: int, min_coverage: float, hp: VoiceEncConfig) -> tuple[int, int]:
    assert n_frames > 0
    win_size = hp.ve_partial_frames
    n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
    
    if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
        n_wins += 1
        
    target_n = win_size + step * (n_wins - 1)
    return n_wins, target_n


def get_frame_step(overlap: float, rate: Optional[float], hp: VoiceEncConfig) -> int:
    assert 0 <= overlap < 1
    if rate is None:
        frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
    else:
        frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
        
    assert 0 < frame_step <= hp.ve_partial_frames
    return frame_step


def stride_as_partials(
    mel: np.ndarray,
    hp: VoiceEncConfig,
    overlap: float = 0.5,
    rate: Optional[float] = None,
    min_coverage: float = 0.8,
) -> np.ndarray:
    """Takes unscaled mels in (T, M) format and slices them into overlapping chunks."""
    assert 0 < min_coverage <= 1
    frame_step = get_frame_step(overlap, rate, hp)
    n_partials, target_len = get_num_wins(len(mel), frame_step, min_coverage, hp)

    if target_len > len(mel):
        padding = np.zeros((target_len - len(mel), hp.num_mels), dtype=mel.dtype)
        mel = np.concatenate((mel, padding), axis=0)
    elif target_len < len(mel):
        mel = mel[:target_len]

    mel = np.ascontiguousarray(mel, dtype=np.float32)
    
    # Safe and readable replacement for manual as_strided
    windows = sliding_window_view(mel, window_shape=(hp.ve_partial_frames, hp.num_mels))
    partials = windows[::frame_step, 0, :, :]
    
    return partials[:n_partials]


class VoiceEncoder(nn.Module):
    def __init__(self, hp: VoiceEncConfig = VoiceEncConfig()):
        super().__init__()
        self.hp = hp

        self.lstm = nn.LSTM(
            self.hp.num_mels, 
            self.hp.ve_hidden_size, 
            num_layers=3, 
            batch_first=True
        )
        if hp.flatten_lstm_params:
            self.lstm.flatten_parameters()
            
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)

        self.similarity_weight = nn.Parameter(torch.tensor([10.0]), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]), requires_grad=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, mels: Tensor) -> Tensor:
        """Computes L2-normalized embeddings of a batch of partial utterances."""
        if self.hp.normalized_mels and (mels.min() < 0 or mels.max() > 1):
            raise ValueError(f"Mels outside. Min={mels.min()}, Max={mels.max()}")

        _, (hidden, _) = self.lstm(mels)
        raw_embeds = self.proj(hidden[-1])
        
        if self.hp.ve_final_relu:
            raw_embeds = F.relu(raw_embeds)
            
        return F.normalize(raw_embeds, p=2, dim=1)

    def inference(
        self, 
        mels: Tensor, 
        mel_lens: Union[Tensor, list], 
        overlap: float = 0.5, 
        rate: Optional[float] = None, 
        min_coverage: float = 0.8, 
        batch_size: Optional[int] = None
    ) -> Tensor:
        """Computes embeddings of a batch of full utterances via sequential chunk sliding."""
        mel_lens_list = mel_lens.tolist() if isinstance(mel_lens, Tensor) else mel_lens
        frame_step = get_frame_step(overlap, rate, self.hp)
        
        n_partials, target_lens = zip(*(
            get_num_wins(l, frame_step, min_coverage, self.hp) for l in mel_lens_list
        ))

        # Pad time-axis if needed
        len_diff = max(target_lens) - mels.size(1)
        if len_diff > 0:
            pad = torch.zeros((mels.size(0), len_diff, self.hp.num_mels), dtype=torch.float32, device=mels.device)
            mels = torch.cat((mels, pad), dim=1)

        # Vectorized generation of overlapping windows using unfold instead of slow Python loops
        win_size = self.hp.ve_partial_frames
        unfolded = mels.unfold(dimension=1, size=win_size, step=frame_step)  # (B, N_wins, M, Win_Size)
        unfolded = unfolded.permute(0, 1, 3, 2)  # (B, N_wins, Win_Size, M)

        # Dynamically isolate accurate sliding windows matching calculated lengths per batch entry
        partials_list = [unfolded[i, :n_p] for i, n_p in enumerate(n_partials)]
        partials = torch.cat(partials_list, dim=0)

        # Compute embedding updates across chunk segments
        n_chunks = int(np.ceil(len(partials) / (batch_size or len(partials))))
        partial_embeds = torch.cat([self(batch) for batch in partials.chunk(n_chunks)], dim=0).cpu()

        # Group and pool segmented features back into singular sequence structures
        slices = np.concatenate(([0], np.cumsum(n_partials)))
        raw_embeds = torch.stack([
            torch.mean(partial_embeds[start:end], dim=0) for start, end in zip(slices[:-1], slices[1:])
        ])
        
        return F.normalize(raw_embeds, p=2, dim=1)

    @staticmethod
    def utt_to_spk_embed(utt_embeds: np.ndarray) -> np.ndarray:
        assert utt_embeds.ndim == 2
        mean_embed = np.mean(utt_embeds, axis=0)
        return mean_embed / np.linalg.norm(mean_embed, ord=2)

    @staticmethod
    def voice_similarity(embeds_x: np.ndarray, embeds_y: np.ndarray) -> float:
        embeds_x = embeds_x if embeds_x.ndim == 1 else VoiceEncoder.utt_to_spk_embed(embeds_x)
        embeds_y = embeds_y if embeds_y.ndim == 1 else VoiceEncoder.utt_to_spk_embed(embeds_y)
        return float(embeds_x @ embeds_y)

    def embeds_from_mels(
        self, 
        mels: Union[Tensor, List[np.ndarray]], 
        mel_lens: Optional[Union[Tensor, list]] = None, 
        as_spk: bool = False, 
        batch_size: int = 32, 
        **kwargs
    ) -> np.ndarray:
        if isinstance(mels, list):
            mels = [np.asarray(mel) for mel in mels]
            assert all(m.shape[1] == mels[0].shape[1] for m in mels), "Mels aren't in (B, T, M) format"
            mel_lens = [mel.shape[0] for mel in mels]
            mels = pack(mels)

        with torch.inference_mode():
            utt_embeds = self.inference(
                mels.to(self.device), mel_lens, batch_size=batch_size, **kwargs
            ).numpy()
            
        return self.utt_to_spk_embed(utt_embeds) if as_spk else utt_embeds

    def embeds_from_wavs(
        self, 
        wavs: List[np.ndarray], 
        sample_rate: int, 
        as_spk: bool = False, 
        batch_size: int = 32, 
        trim_top_db: Optional[float] = 20, 
        **kwargs
    ) -> np.ndarray:
        if sample_rate != self.hp.sample_rate:
            wavs = [
                librosa.resample(wav, orig_sr=sample_rate, target_sr=self.hp.sample_rate, res_type="kaiser_fast")
                for wav in wavs
            ]
            
        if trim_top_db is not None:
            wavs = [librosa.effects.trim(wav, top_db=trim_top_db)[0] for wav in wavs]
            
        kwargs.setdefault("rate", 1.3)
        mels = [melspectrogram(w, self.hp).T for w in wavs]
        
        return self.embeds_from_mels(mels, as_spk=as_spk, batch_size=batch_size, **kwargs)
