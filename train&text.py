NAME = '16_32_d20_acstep4_warm10_CLS'

# LoraConfig
R_VALUE = 4
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TASK_TYPE = "CLS"
TARGET_MODULES= ["q_proj", "k_proj", "v_proj", 
                 "o_proj ", "gate_proj", "up_proj" , 
                 "down_proj", "lm_head"]

# TrainingArguments
NUM_TRAIN_EPOCHS = 8
PER_DEVICE_TRAIN_BATCH_SIZE = 4

DATASET_NUMBER = 6

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
import pandas as pd
def train():
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())

    # 載入模型和Tokenizer
    model_name="taideLlama3_TAIDE_LX_8B_Chat_Alpha1"

    # 量化成4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True
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
    dataset_name="data/"
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
        r = R_VALUE, #要自己調，每個訓練集不一樣.
        lora_alpha = LORA_ALPHA, #r的2倍
        lora_dropout = LORA_DROPOUT,
        bias = "none",
        task_type = TASK_TYPE,
        #  seq_cls會考慮詞出現的順序，cls不會
        target_modules = TARGET_MODULES
        #target_modules = ["q_proj" , "k_proj" , "v_proj" , "o_proj " , "gate_proj" , "up_proj" , "down_proj" , "lm_head"]
        #只有embed_tokens被動到
        #不一定要全選
    )
    # optim : str or [training_args.OptimizerNames], optional, defaults to "adamw_torch"
    # The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor.

    training_arguments = TrainingArguments(
        # here
        output_dir = f"./results_t{DATASET_NUMBER}_{NAME}",
        num_train_epochs = NUM_TRAIN_EPOCHS, 
        bf16 = True, #True對GPU要求較高
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,
        max_grad_norm = 1,
        learning_rate = 1.35e-3, #有特別調小
        weight_decay = 0.001, # 0.001->0.1
        optim = "paged_adamw_8bit",
        #optim = "adamw_hf",
        max_steps = -1, # -1表示沒有限制
        #warmup_ratio = 0.1,
        group_by_length = False, #用來提高訓練效率
        logging_steps = 2,
        lr_scheduler_type = "linear",
        report_to = "wandb",
        save_strategy = "epoch",  # 每個epoch儲存一次
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
    print_trainable_parameters(model)
    wandb.finish()
def merge():
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
    local_model_path = f"results_t{DATASET_NUMBER}_{NAME}/checkpoint-64"
    # try:
    #     local_model_path = "results_t4/epoch3"
    # except:
    #     local_model_path = "results_7_29_/checkpoint-322"
        
    new_model = PeftModel.from_pretrained(base_model, local_model_path)

    # 合并模型權重並卸載適配器
    merged_model = new_model.merge_and_unload()

    # 保存合並後的模型
    merged_model.save_pretrained(f"merged_t{DATASET_NUMBER}_{NAME}")
def test():
    def response(user_input, model, tokenizer):
        # 确保设置 tokenizer 的 pad_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        device = "cuda:0"
        system_prompt = ""
        B_INST, E_INST = "### Instruction:\n", "### Response:\n"
        prompt = f"{system_prompt}{B_INST}{user_input}\n\n{E_INST}"
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # 使用 model.generate 生成结果
        output_ids = model.generate(**inputs, max_new_tokens=248, pad_token_id=tokenizer.pad_token_id)
        # 将生成的 token ids 解码为文字
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 提取 response 部分的文本
        response_text = output_text.split(E_INST)[-1].strip()
        return response_text

    
    base_model_path = "taideLlama3_TAIDE_LX_8B_Chat_Alpha1"
    merged_model_path = f"30_model"
    
    # 量化成4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = False
    )

    # 加载基础模型
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_path,
    #     quantization_config = bnb_config,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.float16,
    #     device_map={"": 0}
    # )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # 加载合并后的模型
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        quantization_config = bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    
    # 讀取 xlsx 檔案
    df_val = pd.read_excel("validation_data/val_6.xlsx")
    df_train = pd.read_excel("repo/train_data_5_text.xlsx")
    start = "請閱讀文章後根據文章主旨以後面提供的標準將文章進行分類：如果文章的內容完全不涉及地層下陷或地面下陷問題，請歸類為「非描述地層下陷問題之文章」。如果文章涉及施工導致的事件，例如路面塌陷、工地事故，或是關於施工過程中的損害處理、安全管理措施、修復策略或事後的應對和改進措施，請歸類為「工地災害：損害處理、安全管理、復修策略、後續回應」。如果文章討論的是地層下陷問題，但這些問題不是由施工安全問題導致的，而是由自然事件（如地震、大雨沖刷、地基塌陷等）造成，或是起因不明的下陷事件，或是對這類事件(非施工相關)的後續應對和改進措施，請歸類為「非施工安全導致之地層下陷問題、後續及其應對措施」 ： "
    
    # 針對每個 input 生成回答
    base_outputs = []
    merged_outputs = []
    
    num=0
    for input_text in df_val["input"]:
        num+=1
        merged_output = response(start+input_text, merged_model, base_tokenizer)
        print(num,":")
        print(merged_output)

        # base_outputs.append(base_output)
        merged_outputs.append(merged_output)
    # 添加生成的回答到 DataFrame
    # df["base_output"] = base_outputs
    df_val["merged_output"] = merged_outputs
    # 保存更新後的 xlsx 檔案
    df_val.to_excel(f"validation_t{DATASET_NUMBER}_{NAME}.xlsx", index=False)
    df_val.to_excel(f"validation_t{DATASET_NUMBER}_{NAME}.xlsx", index=False)
    # ---------------------
    num=0
    merged_outputs = []
    # for input_text in df_train["input"]:
    #     num+=1
    #     merged_output = response(start+input_text, merged_model, base_tokenizer)
    #     print(num,":")
    #     print(merged_output)

    #     # base_outputs.append(base_output)
    #     merged_outputs.append(merged_output)
    # # 添加生成的回答到 DataFrame
    # # df["base_output"] = base_outputs
    # df_train["merged_output"] = merged_outputs
    # # 保存更新後的 xlsx 檔案
    # df_train.to_excel(f"train_quiz.xlsx", index=False)
if __name__ == "__main__":
    # train()
    # merge()
    test()