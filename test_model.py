# pip install -i https://pypi.org/simple/ bitsandbytes
# pip install accelerate
# pip install pyarrow==14.0.1
# pip install requests==2.31.0
# pip install datasets==2.19.0
# pip install peft
# pip install trl
# pip install wandb
# pip install peft

# base模型測試
import pandas as pd
import torch
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
)

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

if __name__ == "__main__":
    base_model_path = "taideLlama3_TAIDE_LX_8B_Chat_Alpha1"
    merged_model_path = "merged_t5_epoch5"
    
    # 量化成4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = False
    )

    #加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config = bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # 加载合并后的模型
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        quantization_config = bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    
    # # 用來看原模型跟訓練完的模型哪些參數不同
    # for (name1, param1), (name2, param2) in zip(base_model.named_parameters(), merged_model.named_parameters()):
    #     if name1 != name2 or not torch.allclose(param1, param2):
    #         print(f"Difference found in parameter: {name1}")
    #         break
    # else:
    #     print("No differences found in parameters.")


    # print("Base Model Config:", base_model.config)
    # print("Merged Model Config:", merged_model.config)
    
    # 測試單篇回應
    # start = "以下會提供一篇可能有提到關鍵字\"地層下陷\"的文章，閱讀以下文章，盡可能仔細地回答以下問題。說明這起有提到\"地層下陷\"的事件的起因是甚麼?是自然發生的還是人為的?你又是從哪些關鍵字句下去判斷的?\n文章："
    # user_input = "跟地層下陷事件相關的基礎設施損壞可能是說哪些基礎設施?"
    
    # print("------------------\nbase_model: \n------------------")
    # print(response(user_input, base_model, base_tokenizer))
    # print("\n\n\n")
    # print("------------------\nmerged_model: \n------------------")
    # print(response(user_input, merged_model, base_tokenizer))
    

    # 讀取 xlsx 檔案
    df = pd.read_excel("validation_data/val_5.xlsx")

    # 針對每個 input 生成回答
    base_outputs = []
    merged_outputs = []
    start = "請閱讀文章後根據文章主旨以後面提供的標準將文章進行分類：如果文章描述的地點是台灣以外的國家，請歸類為「國際新聞」。如果文章涉及施工導致的事件，例如路面塌陷、工地事故，並且重點在於施工過程中的損害處理、安全管理措施、修復策略或事後的應對和改進措施，請歸類為「工地災害：損害處理、安全管理、復修策略、後續回應」。如果文章的內容完全不涉及地層下陷或地面下陷問題，請歸類為「非描述地層下陷問題之文章」。如果文章討論的是地層下陷問題，但這些問題不是由施工安全問題導致的，而是由自然事件（如地震、大雨沖刷、地基塌陷等）造成，或是起因不明的下陷事件，包括文章對這些問題的後續應對和改進措施，請歸類為「非施工安全導致之地層下陷問題、後續及其應對措施」 ： "
    num=0
    for input_text in df["input"]:
        num+=1
        print(num)
        # base_output = response(start+input_text, base_model, base_tokenizer)
        # print(base_output)
        # print("---------------------------------------------")
        merged_output = response(start+input_text, merged_model, base_tokenizer)
        print(merged_output)

        # base_outputs.append(base_output)
        merged_outputs.append(merged_output)
    
    # 添加生成的回答到 DataFrame
    # df["base_output"] = base_outputs
    df["merged_output"] = merged_outputs

    # 保存更新後的 xlsx 檔案
    df.to_excel("validation_t5_epoch5.xlsx", index=False)
    