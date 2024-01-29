# cog-qrcode-lora 
A copy of https://github.com/andreasjansson/cog-qrcode

# Push model to Replicate
https://replicate.com/docs/guides/push-a-model  

## install & login COG on Mac
```angular2html
$ brew install cog
$ cog login
```

## install and start Docker
https://docs.docker.com/get-docker/
  
## download weights
```angular2html
# 1、download lora models to './hf_cache' dir 
https://www.liblib.art/modelinfo/62ed0db76dc34cc3b5779a75c58a6d41
https://www.liblib.art/modelinfo/c7c60c35dc7e4fc09cfa99ddfd5e365b
https://www.liblib.art/modelinfo/dba95d73ac5d4189bf1f12f2394fc727

# 2、download base models to './hf_cache' dir
$ python script/download_weights.py

# or use cog to download base models? (not verify)
$ cog run script/download-weights
```

## build & push
```angular2html
# create model space first at https://replicate.com/create and then push
$ cog push r8.im/<your-username>/<your-model-name>
# e.g. cog push r8.im/datamonet/cog-qrcode-lora
```  

## About test lora  
```angular2html
lora1
beasts cat.
[Trigger-word: JKBB]
https://www.liblib.art/modelinfo/62ed0db76dc34cc3b5779a75c58a6d41

lora2
Guofeng
[no triggrt-word]
https://www.liblib.art/modelinfo/c7c60c35dc7e4fc09cfa99ddfd5e365b

lora3
3ddianshang.
[Trigger-word: 3ddianshang\(style\)]
https://www.liblib.art/modelinfo/dba95d73ac5d4189bf1f12f2394fc727
```

## error 
https://github.com/replicate/cog/issues/1294

## other
peft support multi-lora for DiffusionPipeline, but not StableDiffusionControlNetPipeline
https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference
