from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class Model:
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    
    def get_hidden_state(self, prompt, layer_index):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        # print(hidden_states[layer_index].shape)
        # Extract the hidden state of the last token from the desired layer
        return hidden_states[layer_index][:, -1, :].detach().cpu()

    def predict(self, prompt,length=10):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
