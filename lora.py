import torch
from datasets import load_dataset
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
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os,wandb

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

# 載入模型和Tokenizer
model_name="models/merged_model"
dataset_name="data/"

# 量化成4bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = False
)

# 加載模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map = {"":0}
)

model = prepare_model_for_kbit_training(model)

# 加載tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    add_bos_token=True, 
)
tokenizer.pad_token = tokenizer.eos_token

# 加載數據集
dataset = load_dataset(dataset_name, split = "train")
#print(dataset["instruction"][0])

# 設置wandb
wandb.login(key = "95a5efe7670b48139b6aa90f59088a70ab60766e")
run = wandb.init(
    project = "fine_tune",
    job_type = "training"
)

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _,param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"訓練參數量 : {trainable_params}, 全部參數量 : {all_params}, 訓練參數量占比% : {100*(trainable_params/all_params):.2f}")

# Lora和訓練超參的設定
peft_config = LoraConfig(
    r = 4, #要自己調，每個訓練集不一樣.
    lora_alpha = 8, #r的2倍
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules= ["embed_tokens"]
    #target_modules= ["q_proj" , "k_proj" , "v_proj" , "o_proj " , "gate_proj" , "up_proj" , "down_proj" , "embed_tokens" , "lm_head"]
    #只有embed_tokens被動到
    #不一定要全選
)

training_arguments = TrainingArguments(
    # Output directory where the results and checkpoint are stored
    output_dir = "./results_7_20",
    # Number of training epochs - how many times does the model see the whole dataset
    num_train_epochs = 7, #Increase this for a larger finetune
    # Enable fp16/bf16 training. This is the type of each weight. Since we are on an A100
    # we can set bf16 to true because it can handle that type of computation
    bf16 = False,#True對GPU要求較高
    # Batch size is the number of training examples used to train a single forward and backward pass. 
    per_device_train_batch_size = 2,
    # Gradients are accumulated over multiple mini-batches before updating the model weights. 
    # This allows for effectively training with a larger batch size on hardware with limited memory
    gradient_accumulation_steps = 2,
    # memory optimization technique that reduces RAM usage during training by intermittently storing 
    # intermediate activations instead of retaining them throughout the entire forward pass, trading 
    # computational time for lower memory consumption.
    gradient_checkpointing = True,
    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 1,
    # Initial learning rate (AdamW optimizer)
    learning_rate = 5e-6, #有特別調小
    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001,
    # Optimizer to use
    optim = "paged_adamw_8bit",
    # Number of training steps (overrides num_train_epochs)
    max_steps = -1,# -1表示沒有限制
    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03,
    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True, #用來提高訓練效率
    # Log every X updates steps
    logging_steps = 30,
    lr_scheduler_type = "linear",
    report_to = "wandb"
)

# 模型微調
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    peft_config = peft_config,
    tokenizer = tokenizer,
    dataset_text_field = "text",
    args = training_arguments,
    packing = False
)

trainer.train()
model = get_peft_model(model, peft_config) # 獲取微調後的模型
print_trainable_parameters(model)
trainer.model.save_pretrained("/7_18/final_model/")
wandb.finish()
# model.config.use_cache = True # 啟用模型的緩存功能，這樣在進行推理時可以提高計算效率和速度。
# model.eval() #  將模型設置為評估模式。這會關閉訓練過程中的一些特定功能（如dropout），以確保模型在推理時能夠穩定和一致地進行預測。