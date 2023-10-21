import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import time

start = time.time()
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("model")
end = time.time()
print(f"1__{end - start:.5f} sec")


start = time.time()
file = "소아남여_소아남여01_F_1592580488-0_7_수도권_실내_00466.wav"
arr, sampling_rate = librosa.load(file, sr=16000)
print(type(arr))

end = time.time()
print(f"2__{end - start:.5f} sec")

start = time.time()
input_features = processor(arr, return_tensors="pt", sampling_rate=sampling_rate).input_features
end = time.time()
print(f"3__{end - start:.5f} sec")

start = time.time()
forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
end = time.time()
print(f"4__{end - start:.5f} sec")

print(transcription)

