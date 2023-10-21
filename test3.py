import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import boto3
import requests
import subprocess
from flask import Flask, request
from typing import Any
import sys
import time
import openai
import datetime
import re
import numpy as np

import parmap
import multiprocessing
from multiprocessing import freeze_support

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import settings
# AWS 계정 정보 및 S3 버킷 정보 설정
aws_access_key_id = 'AKIAUKTRTP4V7HMI3BPK'
aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
bucket_name = 'seocho-voicetest'

# S3 클라이언트 생성
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# openai api key
api_key = settings.OPENAI_KEY
openai.api_key = api_key

# stt model
stt_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
stt_model = WhisperForConditionalGeneration.from_pretrained("seocho/model")

# stable diffusion modelz
# sd_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
# sd_model.to("cuda")


#sd_model = pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
sd_model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
sd_model = sd_model.to("cuda")


# s3에서 webm 데이터 받아온 후 wav 파일로 변환
def get_webm(s3_audio_url, webm_path):
    try:
        response = requests.get(s3_audio_url)
        if response.status_code == 200:
            with open(webm_path, 'wb') as local_file:
                local_file.write(response.content)
            print(f"오디오 파일이 다운로드되었습니다. 로컬 파일 경로: {webm_path}")
        else:
            print(f"다운로드 중 오류가 발생했습니다. 응답 코드: {response.status_code}")
    except Exception as e:
        print(f"다운로드 중 오류가 발생했습니다: {str(e)}")   
        
def webm_to_wav(filename, wav_path):
    ffmpeg_command = [
    'ffmpeg',  # ffmpeg 실행 명령어
    '-i', filename,  # 입력 파일 경로
    '-vn',  # 비디오 스트림 무시
    '-acodec', 'pcm_s16le',  # 오디오 코덱 설정 (16-bit PCM)
    '-ar', '16000',  # 샘플링 레이트 설정 (예: 44100 Hz)
    '-ac', '1',  # 채널 수 설정 (2 채널 = 스테레오)
    wav_path  # 출력 파일 경로
    ]
    subprocess.run(ffmpeg_command,stdout=subprocess.PIPE,stdin=subprocess.PIPE)
        

# 음성 인식
def stt(audio_file):

    arr, sampling_rate = librosa.load(audio_file, sr=16000)

    input_features = stt_processor(arr, return_tensors="pt", sampling_rate=sampling_rate).input_features

    forced_decoder_ids = stt_processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    predicted_ids = stt_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription)
    return transcription


# Prompt Engineering
def load_template(type):
    # UnicodeDecodeError로 encoding='utf-8'추가
    with open('seocho/prompt/prompt_template'+type+'.txt', 'r', encoding='utf-8') as f:
        return f.read()
    
def make_gpt_format(query: str = None):
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    return message

def gpt3(
        query: str = None,
        model_name: str = "gpt-4",
        max_len: int = 3500,
        temp: float = 1.0,
        n: int = 1,
        freq_penalty: float = 0.0,
        pres_penalty: float = 0.0,
        top_p: float = 1.0,
        type: str = '1',
    ):
    if not query:
        raise ValueError("There is no query sentence!")
    prompt_template = load_template(type)
    
    prompt = prompt_template.format(transcribed_text=query)
    
    messages = make_gpt_format(prompt)
    # print(messages)
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=max_len,
                temperature=temp,
                n=n,
                presence_penalty=pres_penalty,
                frequency_penalty=freq_penalty,
                top_p=top_p,
            )
            received = True
        except openai.error.OpenAIError as e:
            print(f"OpenAIError: {e}.")
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g., prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{messages}\n\n")
                assert False
            time.sleep(2)
    resp = response.choices[0].message["content"]
    
    print(resp)
    return resp

# Stable Diffusion
def text_to_image(text):
    print('start')
    #prompt = "a photograph of an astronaut riding a horse"
    image = sd_model(text).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
    print('done')
    #이미지 저장
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S')
    image_file_path = 'seocho/image/image'+formatted_time+'.jpg'
    image.save(image_file_path)
    
    try:
        s3.upload_file(image_file_path, bucket_name, image_file_path)
        print(f"{image_file_path} 파일이 {bucket_name} 버킷에 성공적으로 업로드되었습니다.")
    except Exception as e:
        print(f"업로드 중 오류가 발생했습니다: {str(e)}")
    
    url = 'https://seocho-voicetest.s3.ap-northeast-2.amazonaws.com/'+image_file_path

    #이미지 출력
    return url


    
def main(s3_audio_url):
    
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M%S')
    
    print(formatted_time)
    '''
    # WebM 파일을 저장할 경로
    webm_file_path = 'seocho/audio/audio'+formatted_time+'.webm'

    # S3 URL에서 오디오 파일 다운로드
    get_webm(s3_audio_url, webm_file_path)   

    # wav 파일을 저장할 경로
    wav_file_path = 'audio'+formatted_time+'.wav'
    
    # webm 파일을 wav 파일로 변환
    webm_to_wav(webm_file_path, wav_file_path)
    '''
    # 임시로 audio 값 고정
    wav_file_path = "seocho/audio/audio20230930161634.wav"

    # speech to text
    text = stt(wav_file_path)
    text = '상어'
    
    # 페이지 수 설정
    num = 5
    
    # 동화책 내용 생성
    story = gpt3(query=text, type='4')
    
    page_sentences = []

    # 페이지별로 문장 추출
    pages = story.split('\n')
    for page in pages:
        if page.strip():  # 빈 줄은 무시
            # 페이지에서 문장만 추출
            sentence = page.split(': ')[1]
            page_sentences.append(sentence)
    
    #print(page_sentences) 



    title = gpt3(query=story, type='3')
    
    # 정규 표현식 패턴
    pattern = r'제목:\s+"([^"]+)"'

    # 정규 표현식을 사용하여 제목 추출
    match = re.search(pattern, title)

    if match:
        title = match.group(1)
        print(f"제목: {title}")
    else:
        print("제목을 찾을 수 없습니다.")

    prompt_result = gpt3(query=story, type='5')
    prompt_list = []
    
    prompts = prompt_result.split('\n')
    for prompt in prompts:
        if prompt.strip():  # 빈 줄은 무시
            # 페이지에서 문장만 추출
            text = prompt.split(': ')[1]
            prompt_list.append(text)
            
    print(prompt_list)
    
    color = ['painted in bright water colors']
    
    medium = ['colored pencil', 'oil pastel', 'acrylic painting', 'a color pencil sketch inspired by Edwin Landseer', 'large pastel, a color pencil sketch inspired by Edwin Landseer', 'a storybook illustration by Marten Post', 'naive art', 'cute storybook illustration', 'a storybook illustration by tim burton']
    
    setting = ","+ medium[0] +", (Lighting) in warm sunlight, (Artist) in a cute and adorable style, cute storybook illustration, (Medium) digital art, (Color Scheme) vibrant and cheerful, "
    
    # text to image
    result_img = []
    for i in range(num):
        print(prompt_list[i]+setting)
        result_img.append(text_to_image(prompt_list[i]+setting))
        
    # s3 image url 반환
    return 0


if __name__ == "__main__":
    main("https://seocho-voicetest.s3.ap-northeast-2.amazonaws.com/audio-1696055947410.webm")