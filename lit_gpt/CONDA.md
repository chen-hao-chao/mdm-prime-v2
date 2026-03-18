
```sh
conda create -n smdm python=3.9
conda activate smdm

pip install torch torchvision torchaudio

# install flash-attention
pip uninstall ninja -y && pip install ninja -U

git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
pip install packaging
python setup.py install

cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../../.. && rm -r flash-attention

# install xformers
pip install -U xformers

# install TinyLama requirements
git clone https://github.com/jzhang38/TinyLlama.git
cd TinyLlama
pip install -r requirements.txt tokenizers sentencepiece
cd .. && rm -r TinyLlama

# Install the dependencies needed for evaluation
pip install lm-eval==0.4.4 numpy==1.25.0 bitsandbytes==0.43.1
pip install openai==0.28 fschat==0.2.34 anthropic
```