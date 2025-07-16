# SayCan 로봇 행동 계획 시스템 

본 프로젝트는 2022년 Google에서 제안한 언어 기반 로봇 행동 계획 알고리즘인 **SayCan**을 재구성한 Jupyter Notebook입니다. 기존 SayCan은 대형 언어모델(GPT)과 로봇의 행동 가능성 정보(Affordance)를 결합하여 자연어 명령을 실제 로봇 동작으로 변환하는 구조를 가지고 있습니다.

기타자료는 노션의 "혼자 공부하는 SAYCAN" 참고

---

## 🔧 프로젝트 개요 및 수정 내역

SayCan의 원본 코드를 실행하는 과정에서 다음과 같은 주요 문제들이 발생:

- 2022년 기준으로 작성된 코드가 **구버전 라이브러리에 의존**함  
- PyTorch, TensorFlow, JAX, Flax, Transformers, OpenAI API 등의 **인터페이스 변경 또는 함수 제거**
- GPT 모델 호출 방식 변경으로 인한 **API 불호환 문제**

이에 따라 본 프로젝트에서는 전체 코드를 다음과 같이 개편:

- 더 이상 지원되지 않는 구문 제거 및 최신 문법 적용
- `transformers`, `jax`, `flax`, `optax`, `openai` 등 **최신 버전에 맞게 호환성 확보**
- GPT-3.5 기반의 OpenAI API로 **정밀도 및 안정성 개선**
- **NVIDIA RTX A6000 GPU 서버** 환경에 맞춰 JAX 및 CUDA 설정을 재정비

---
## 🔐 OpenAI API 키 사용 관련 주의사항

본 프로젝트는 OpenAI GPT API를 사용합니다.  
반드시 **본인의 OpenAI API 키**를 사용해야 하며,  
**GitHub 보안 정책에 따라 코드에서 제외하였습니다.**

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

---
## ⚙️ 개발 환경 및 설치 방법

### 🐍 Conda 가상환경 설정 및 패키지 설치

```bash
conda create -n saycans python=3.9 -y
conda activate saycans
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9

pip install jinja2 pyyaml typeguard
pip install tensorflow==2.11.0

pip install jax[cuda11_cudnn86]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.6.11 optax==0.1.5 chex==0.1.7

pip install openai easydict tqdm requests ftfy regex
pip install git+https://github.com/openai/CLIP.git
pip install tensorboard

conda install -c conda-forge moviepy -y
pip install moviepy==1.0.3 imageio==2.9.0

pip install numpy==1.26.4

pip install jax==0.4.27 jaxlib==0.4.27 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.8.2 optax==0.1.7 orbax-checkpoint==0.6.4

pip install scipy==1.10.1
pip install pybullet

pip install --upgrade "jax[cuda11_pip]>=0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade flax optax chex

pip install opencv-python
pip install matplotlib
pip install charset_normalizer
