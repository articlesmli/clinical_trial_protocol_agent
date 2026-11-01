# scripts/run_sft_training.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

def run_fine_tuning():
    # 1. Load your generated clinical trial jsonl dataset
    dataset_path = "data/fine_tuning/clinical_criteria_sft.jsonl"
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    print(f"[Training] Loaded {len(dataset)} training records from {dataset_path}")

    # 2. Choose a lightweight base model (e.g., a small open-source instruction model)
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct" # or any compatible causal LM
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # 3. Configure PEFT/LoRA for memory-efficient training
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # Varies depending on model architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Set up Training Arguments
    training_args = TrainingArguments(
        output_dir="./clinical_model_checkpoint",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=3,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
    )

    # 5. Initialize TRL's SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    print("[Training] Starting model fine-ning loop...")
    trainer.train()
    
    # 6. Save your newly fine-tuned clinical model domain expert
    trainer.model.save_pretrained("./fine_tuned_clinical_model")
    tokenizer.save_pretrained("./fine_tuned_clinical_model")
    print("[Training] Model successfully fine-tuned and saved to ./fine_tuned_clinical_model!")

if __name__ == "__main__":
    run_fine_tuning()
