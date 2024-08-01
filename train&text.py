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
        r = 2, #要自己調，每個訓練集不一樣.
        lora_alpha = 4, #r的2倍
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CLS",
        #  seq_cls會考慮詞出現的順序，cls不會
        target_modules= ["q_proj" , "k_proj" , "v_proj" , "o_proj " , "gate_proj" , "up_proj" , "down_proj" , "lm_head"]
        #target_modules= ["q_proj" , "k_proj" , "v_proj" , "o_proj " , "gate_proj" , "up_proj" , "down_proj" , "lm_head"]
        #只有embed_tokens被動到
        #不一定要全選
    )

    training_arguments = TrainingArguments(
        output_dir = "./results_t5_no_embed",
        num_train_epochs = 5, 
        bf16 = False,#True對GPU要求較高
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        gradient_checkpointing = True,
        max_grad_norm = 0.5,
        learning_rate = 2e-5, #有特別調小
        weight_decay = 0.001,## 0.001->0.1
        optim = "paged_adamw_8bit",
        max_steps = -1,# -1表示沒有限制
        warmup_ratio = 0.03,
        group_by_length = True, #用來提高訓練效率
        logging_steps = 10,
        lr_scheduler_type = "linear",
        report_to = "wandb",
        save_strategy = "epoch",  # 每個epoch儲存一次
    )
    #

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
    local_model_path = "results_t4/epoch5"
    # try:
    #     local_model_path = "results_t4/epoch3"
    # except:
    #     local_model_path = "results_7_29_/checkpoint-322"
        
    new_model = PeftModel.from_pretrained(base_model, local_model_path)

    # 合并模型權重並卸載適配器
    merged_model = new_model.merge_and_unload()

    # 保存合並後的模型
    merged_model.save_pretrained("merged_t4_epoch5")
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
        output_ids = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.pad_token_id)
        # 将生成的 token ids 解码为文字
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 提取 response 部分的文本
        response_text = output_text.split(E_INST)[-1].strip()
        return response_text

    
    base_model_path = "taideLlama3_TAIDE_LX_8B_Chat_Alpha1"
    merged_model_path = "merged_t4_epoch5"
    
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
    df = pd.read_excel("validation_data/val_2.xlsx")

    # 針對每個 input 生成回答
    # base_outputs = []
    merged_outputs = []
    start = "請根據輸入的這篇新聞文章的內容，判斷其屬於哪一種類別（分別有：「應對措施」(應對措施分類中的文章是在描述為了改善地層下陷相關的問題所採取的行動，可以是政府頒布的一項政策，或是一項企劃)、「事件描述」(事件描述分類中的文章主要是關於各種原因造成的地層下陷事件，可以是講述一個地區所觀測到的地層下陷量，也可以是講述一地區突然出現的未知成因塌陷)、「事件回應」(事件回應分類中的文章是在描述相關人物針對地層下陷事件後續之的回應或是對此事件的評價)、「其他」(被分於其他類別中的文章通常僅提到地層下陷或者並未提及，主要在談論的主題與並非地層下陷。)），請判斷是以上四個類別中的哪一種:："
    num=0
    for input_text in df["input"]:
        num+=1
        merged_output = response(start+input_text, merged_model, base_tokenizer)
        print(num,":")
        print(merged_output)

        # base_outputs.append(base_output)
        merged_outputs.append(merged_output)
    
    # 添加生成的回答到 DataFrame
    # df["base_output"] = base_outputs
    df["merged_output"] = merged_outputs

    # 保存更新後的 xlsx 檔案
    df.to_excel("validation_t4_epoch5.xlsx", index=False)
if __name__ == "__main__":
    train()
    # merge()
    # test()