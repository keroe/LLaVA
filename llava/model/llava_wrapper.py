import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import requests
from io import BytesIO

class LlavaWrapper:
    def __init__(self, model_path, model_base=None):
        disable_torch_init()
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(model_path, model_base, self.model_name)

    def wrap_query_with_tokens(self, query):
        if self.model.config.mm_use_im_start_end:
            query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            query = DEFAULT_IMAGE_TOKEN + '\n' + query
        return query

    def load_image_as_tensor(self, image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    def get_conv_mode(self):
        if 'llama-2' in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        return conv_mode

    def query_to_prompt_and_stop_str(self, query):
        conv_mode = self.get_conv_mode()
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        return conv.get_prompt(), stop_str

    def post_process_output(self, input_ids, output_ids, stop_str):
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        return outputs.strip()

    def run(self, query, image_file):
        query = self.wrap_query_with_tokens(query)
        prompt, stop_str = self.query_to_prompt_and_stop_str(query)
        image_tensor = self.load_image_as_tensor(image_file)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.post_process_output(input_ids, output_ids, stop_str)
        print(outputs)

