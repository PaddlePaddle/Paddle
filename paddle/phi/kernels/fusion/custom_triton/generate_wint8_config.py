
import re

def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


template =  '''
python3.8  ${compile_file}     \
/zhoukangkang/2023-06-06minigpt/PaddleNLP/paddlenlp/experimental/transformers/fused_transformer_layers.py    \
-n matmul_kernel   \
-o ${wint8_dir}/wint8     \
--out-name wint8_kernel     \
-w ${num_warps}   -ns ${num_stages} \
-s   "*fp16:16, *i8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, ${block_m}, ${block_n}, ${block_k}, 1, ${split_k}"\
 -g   "((M+${block_m}-1)/${block_m}) * ((N+${block_n}-1)/${block_n}), ${split_k}, 1" 
'''

for num_stages in [2, 3, 4, 5, 6]:
    for block_m in [16]:
        for block_n in [16, 32, 64, 128]:
            for block_k in [32, 64, 128, 256]:
                num_warps = 4
                if block_n >= 128 and block_k >= 256:
                    continue
                for split_k in [1, 2, 4, 8]:
                    values = {
                        "num_stages": str(num_stages),
                        "block_m": str(block_m),
                        "block_n": str(block_n),
                        "block_k": str(block_k),
                        "split_k": str(split_k),
                        "num_warps": str(num_warps)
                    }
                    print(SubstituteTemplate(template, values))







