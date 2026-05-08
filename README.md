there's anoother nt_models folder you need for the transformer, so i will compress that folder and send it later 


run this in your terminal to install the NT transformer model: 

HF_HUB_DISABLE_XET=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
    local_dir='./nt_model',
)
"

if i do end up compressing the folder, though, then just extract it and put it in this repository. 