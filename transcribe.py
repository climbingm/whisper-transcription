import torch
from transformers import pipeline
import sys

def transcribe(audioFilePath):
    device = 0 if torch.cuda.is_available() else "cpu"
    
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-large-v2",
        chunk_length_s=30,
        device=device,
    )
    
    print("🔄 Transcribing...")
    out = pipe(audioFilePath, return_timestamps=True)
    
    # Save plain text transcription
    with open("Output-Text.txt", "w", encoding="utf-8") as text_file:
        text_file.write(out["text"])
    
    # Save timestamped transcription
    with open("Output-Timestamped.txt", "w", encoding="utf-8") as timestamped_file:
        for chunk in out["chunks"]:
            start, end = chunk["timestamp"]
            timestamped_file.write(f"[{start:.2f} - {end:.2f}] {chunk['text']}\n")
    
    print("✅ Transcriptions saved to Output-Text.txt and Output-Timestamped.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python transcribe.py <audioFilePath>")
        sys.exit(1)
    
    audioFilePath = sys.argv[1]
    transcribe(audioFilePath)