import torch
import torchaudio
import os
import argparse
from tqdm import tqdm
from torio.io import StreamingMediaDecoder, StreamingMediaEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Join audio to video files.")
    parser.add_argument("dataset", type=str, choices=["moviegen", "vggsound"], help="Dataset to process.")
    parser.add_argument("video_idx", type=str, help="Video index to process.")
    return parser.parse_args()


class VideoJoiner:

    def __init__(
        self,
        video_root: str,
        output_root: str,
        sample_rate: int = 16000,
        duration_secs: float = 8.,
    ):
        assert output_root != video_root, "output_root should be different from video_root."
        self.video_root = video_root
        self.output_root = output_root
        self.sample_rate = sample_rate
        self.duration_secs = duration_secs

        os.makedirs(self.output_root, exist_ok=True)

    def read_audio(self, audio_path: str) -> torch.Tensor:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if audio.shape[1] >= self.sample_rate * self.duration_secs:
            audio = audio[:, :int(self.sample_rate * self.duration_secs)]
        elif audio.shape[1] < self.sample_rate * self.duration_secs:
            raise ValueError(
                f"Audio length {audio.shape[1]} is shorter than expected {self.sample_rate * self.duration_secs}."
            )

        return audio

    def join(self, video_id: str, audio_path: str, model_name: str):
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        output_path = os.path.join(self.output_root, video_id, f"{model_name}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        audio = self.read_audio(audio_path)
        add_audio_to_video(video_path, output_path, audio, self.sample_rate, self.duration_secs)


def add_audio_to_video(
    video_path: str,
    output_path: str,
    audio: torch.Tensor,
    sample_rate: int = 16000,
    duration_secs: float = 8.0
):
    frame_rate = 24
    reader = StreamingMediaDecoder(video_path)
    reader.add_basic_video_stream(
        frames_per_chunk=int(frame_rate * duration_secs),
        format="rgb24",
        frame_rate=frame_rate,
    )
    reader.fill_buffer()
    video_chunk = reader.pop_chunks()[0]
    _, _, h, w = video_chunk.shape

    # Ensure audio is in correct format (samples, channels)
    if audio.dim() == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.transpose(0, 1)

    writer = StreamingMediaEncoder(output_path)
    writer.add_audio_stream(
        sample_rate=sample_rate,
        num_channels=audio.shape[1] if audio.dim() == 2 else 1,
        encoder="libmp3lame",
    )
    writer.add_video_stream(
        frame_rate=frame_rate,
        width=w,
        height=h,
        format="rgb24",
        encoder="libx264",
        encoder_format="yuv420p",
    )

    with writer.open():
        writer.write_audio_chunk(0, audio.float())
        writer.write_video_chunk(1, video_chunk)


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "moviegen":
        video_root = "/home/thanhtvt1sc/Downloads/MovieGen-baselines/MovieGenAudioBenchSfx/video_with_audio/"
        output_root = "/home/thanhtvt1sc/Downloads/MovieGen-baselines/MovieGenAudioBenchSfx/video_join/"
        audio_path_dict = {
            "Flowley": f"/home/thanhtvt1sc/Downloads/MovieGen-baselines/flowley_moviegen/2025-07-26_10-30-22_emaiter=0-60000_fmstep=25_cfg=7.5_moviegen/{args.video_idx}.wav",
            "VinTAGe": f"/home/thanhtvt1sc/Downloads/MovieGen-baselines/vintage_moviegen/pred/{args.video_idx}.wav",
            "MovieGenAudio": f"/home/thanhtvt1sc/Downloads/MovieGen-baselines/MovieGenAudioBenchSfx/audio_only/{args.video_idx}.wav",
        }
    else:
        video_root = "/home/thanhtvt1sc/Downloads/VGGSound-baselines/vggsound-test-videos/"
        output_root = "/home/thanhtvt1sc/Downloads/VGGSound-baselines/video_join/"
        audio_path_dict = {
            "Flowley": f"/home/thanhtvt1sc/Downloads/VGGSound-baselines/flowley-cfg75/2025-07-26_10-30-22_emaiter=0-60000_fmstep=25_cfg=7.5/{args.video_idx}.wav",
            "FoleyCrafter": f"/home/thanhtvt1sc/Downloads/VGGSound-baselines/foleycrafter/audio/{args.video_idx}.wav",
            "Frieren": f"/home/thanhtvt1sc/Downloads/VGGSound-baselines/frieren/CFG4.5_euler_26gen_wav_16k_80_last/Y{args.video_idx}_9.wav",
            "Ground-truth": f"/home/thanhtvt1sc/Downloads/VGGSound-baselines/gt/gt/{args.video_idx}.wav",
            "MDSGen": f"/home/thanhtvt1sc/Downloads/VGGSound-baselines/mdsgen/output_wav/{args.video_idx}.wav",
            "Mel-QCD": f"/home/thanhtvt1sc/Downloads/VGGSound-baselines/melqcd/audio/{args.video_idx}.wav",
            "MMAudio": f"/home/thanhtvt1sc/Downloads/VGGSound-baselines/mmaudio/test-sampled/{args.video_idx}.wav",
            "V2A-Mapper": f"/home/thanhtvt1sc/Downloads/VGGSound-baselines/v2a-mapper/v2a-mapper-filtered/{args.video_idx}.wav",
            "VinTAGe": f"/home/thanhtvt1sc/Downloads/VGGSound-baselines/vintage/pred/{args.video_idx}.wav",
        }

    video_joiner = VideoJoiner(video_root, output_root)
    for model_name, audio_path in tqdm(audio_path_dict.items()):
        try:
            if os.path.exists(audio_path):
                video_joiner.join(args.video_idx, audio_path, model_name)
                print(f"Successfully joined {model_name}")
            else:
                print(f"Skipping {model_name}: audio file not found at {audio_path}")
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
