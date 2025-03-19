from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

from pprint import pprint


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pprint(device)

ds = Dataset.load_from_disk("../data/alpaca_data_zh/")
#pprint(ds[:3])

tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")

def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
pprint(tokenizer.decode(tokenized_ds[1]["input_ids"]))
pprint(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"]))))

model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh")

for name, parameter in model.named_parameters():
    print(name)

from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                    target_modules=[
                        "query_key_value",
                    ]
                    )
#pprint(config)

model = get_peft_model(model, config)
model.to(device)
print(model)

#for name, parameter in model.named_parameters():
#    print(name)

model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=100,
    num_train_epochs=2
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

model = model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
pprint(tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True))














