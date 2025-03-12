---
title: Deploying Large Language Models Locally
date: 2025-03-12T17:56:11+05:30
draft: false
image: /images/blog/post-3/banner.webp
meta_title: LLM Local Deployment
tags:
  - AI
  - LLM
categories:
  - AI
  - Python
description: Blog about how to deploy LLM for local use on personal computers
author: Raunak Wete
---

## Introduction

We’ve all used AI chat models like **ChatGPT, Gemini, and Claude** in our daily lives. They’re great, but what happens when you want to **integrate AI into your own applications**? Most people turn to APIs—but there's a catch.

💰 **API costs can add up fast**, especially if you’re a student or working on an experimental project. Testing, debugging, and fine-tuning prompts require **millions of tokens**, which can quickly become **too expensive to sustain**.

But what if I told you there’s a way to do it all **for free**? Yes, you read that right! Instead of paying for every API request, you can **run powerful AI models like Llama, Mistral, and Qwen** **right on your own computer**—without spending a dime.

We will be learning to install *Ollama* on our local machine for easy setup and deployment of AI models.

By the end of this guide, you’ll have a fully functional **AI model running on your own machine**, ready to use **without API costs or internet dependency**. Let’s get started!

## Ollama

**Ollama** is a **lightweight, hassle-free** framework designed to run AI models **directly on your PC**. It’s the **perfect choice** if you want a **one-click setup** without dealing with complex configurations. Whether you’re experimenting with AI, building a personal project, or just want a chatbot on your local machine, **Ollama makes it effortless**.

###  Why Choose Ollama?

1. **Quick & Easy Setup** – No complicated installations, just a simple command.  
2. **Runs on CPU & GPU** – Works on most computers, with GPU acceleration if available.  
3. **Supports Popular Models** – Run Llama 2, Mistral, Qwen, and more with zero hassle.  
4. **Offline & Free** – No API costs, no internet required..

### Installation
##### Windows:
- Open Windows PowerShell and enter this command 
```powershell
Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" `
∙ -OutFile "$env:HOMEPATH\Downloads\ollamaSetup.exe"; `
∙ Start-Process "$env:HOMEPATH\Downloads\ollamaSetup.exe"  -Wait; `
∙ Remove-Item "$env:HOMEPATH\Downloads\ollamaSetup.exe" -Force
```
- Alternatively, you can **manually download** the setup from [Ollama’s official website](https://ollama.com/download) and install it.
##### Mac:
- Head over to [Ollama’s download page](https://ollama.com/download) and grab the **macOS installer**.
##### Linux:
- Open bash/zsh shell and run
```zsh
curl -fsSL https://ollama.com/install.sh | sh
```
- If you’re using **Arch Linux**, install Ollama from the AUR:
```zsh
sudo pacman -Sy ollama # For CPU
sudo pacman -Sy ollama-cuda # For Nvidia GPU
sudo pacman -Sy ollama-rocm # For AMD GPU
```


> *Restart the PC after installation*

### Usage

- Open terminal (PowerShell on Windows) and run
```zsh
ollama serve
```
- This will start the ollama server on [http://localhost:11434](http://localhost:11434)
- By default, Ollama only runs on **localhost**. If you want to access it from other devices on your network, you need to **change the host binding**:
	- Set the environment variable ` OLLAMA_HOST=0.0.0.0 `
	-  On **Windows**, set the environment variable:
    ```pwsh
    $env:OLLAMA_HOST="0.0.0.0" ollama serve
    ```
    
	- On **Linux/macOS**, you can run:
	```zsh
	OLLAMA_HOST=0.0.0.0 ollama serve
	```
    
This will make Ollama accessible across your **entire local network**, allowing other devices to send API requests to your AI model.

- **Download a Model**:
	- Choose a llm model of your choice on [Ollama Models](https://ollama.com/search)
	- Here, we will be using a basic ***Llama3.2:3B*** model.
	```zsh
	ollama pull llama3.2:3b
	```

	- After downloading you can run this model using
	```zsh
	ollama run llama3.2:3b
	```

	- You can list all your downloaded models by running
	```zsh
	ollama list
	```


Now you’re all set! You can start interacting with AI models **without relying on external APIs**—all from your own device.


### Using Ollama API

*Ollama API* is *OpenAI API* compatible which means you can directly use OpenAI python library to interact with Ollama.

This makes migrating your existing applications using OpenAI API to Ollama very easy.

- Install *openai* python package
 ```zsh
pip install openai
```
- Test if the api is working
```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

completion = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn."
        }
    ]
)

print(completion.choices[0].message.content)
```

> *model* parameter must be the model name of the downloaded models. You can find the downloaded models using `ollama list` command.

## Conclusion

Congratulations!  You’ve just learned how to deploy Large Language Models (LLMs) **locally** using **Ollama** --  giving you the power of AI **without API costs or internet dependency**. Whether you’re a hobbyist exploring AI or a developer integrating models into your applications, running LLMs locally opens up **endless possibilities**.

But this is just the beginning! 

We’ll be bringing **more exciting blogs** in the future, covering advanced topics like:  
-  **Deploying AI models for production workloads** – Scaling your AI applications beyond local use.  
-  **Optimizing AI servers for fast performance** – Reducing latency and maximizing efficiency.  
-  **Cutting costs while running powerful AI models** – Getting the most out of your hardware without breaking the bank.

Enjoy your **Holi** running powerful models like ***DeepSeek-R1***, etc on your PC.

💡 **Got questions or ideas for future topics? Comment below!** 👇