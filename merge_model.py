from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    TextStreamer,
    Trainer
)
model_name = "taideLlama3_TAIDE_LX_8B_Chat_Alpha1"

# 載入基礎模型，我這裡是從本地加載，也可以從huggingFace直接複製模型名稱
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0}
)

# 從本地加載適配器
# here
local_model_path = "results_t5_alpha32/checkpoint-100"
new_model = PeftModel.from_pretrained(base_model, local_model_path)

# 合并模型權重並卸載適配器
merged_model = new_model.merge_and_unload()

# 保存合並後的模型
# here
merged_model.save_pretrained("merged_t5_noEmbed_epoch5_alpha32")