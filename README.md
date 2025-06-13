# llmsrv


## build

create venv: `python -m venv venv`

then install requirements: `pip install -r requirements.txt`

then llama.cpp: `CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --upgrade --force-
reinstall --no-cache-dir`

## run

`uvicorn llmsrv:app --reload --port 8080`
