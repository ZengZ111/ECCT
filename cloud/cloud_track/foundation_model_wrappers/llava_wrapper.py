
import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase


class LlavaWrapper(WrapperBase):
    def __init__(
        self,
        model_name="/home/zz/models/llava-1.5-13b-hf",
        enable_caching=True,
        simulate_time_delay=False,
        system_prompt=None,
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Llava model from {model_name} on device {self.device}...")
        #############
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        #quantization_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #    bnb_4bit_compute_dtype=torch.float16,
        #)
        #self.pipe = pipeline(
        #    "image-to-text",
        #    model=model_name,
        #    model_kwargs={"quantization_config": quantization_config},
        #)

        ###############
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"Model loaded from {model_name}")
        self.system_prompt = system_prompt

    def run_inference(self, prompt: str, image: Image):
        if not self.system_prompt:
            raise ValueError("System prompt is not set.")

        prompt_1_final = f"USER: <image>\n {self.system_prompt} ASSISTANT:"

        ans = self.run_inference_inner(prompt_1_final, image)
        answer_1 = ans.split("ASSISTANT:")[-1].strip()

        prompt_2 = f"USER: <image>\n{self.system_prompt} ASSISTANT: {answer_1}</s>USER: {prompt} ASSISTANT:"
        ans = self.run_inference_inner(prompt_2, image)
        answer_2 = ans.split("ASSISTANT:")[-1].strip()

        answer_2_yes_no = answer_2.lower().strip()
        if "yes" in answer_2_yes_no:
            answer_2_yes_no = "yes"
        elif "no" in answer_2_yes_no:
            answer_2_yes_no = "no"

        ans_formatted = (
            f"Answer: {answer_2_yes_no} \nJustification: {answer_2}"
        )

        return ans_formatted

    def run_inference_inner(self, prompt, image):
        inputs = self.processor(text=prompt, images=image,return_tensors = "pt").to(self.device)

        # Generate
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, max_new_tokens=64)

        ans = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

        #ans = self.pipe(
        #    image, prompt=prompt, generate_kwargs={"max_new_tokens": 30}
        #)

        return ans[0]


if __name__ == "__main__":
    system_prompt = "You are an intelligent assistant, helping a drone on a search and rescue mission. Describe the image."
    llava = LlavaWrapper(system_prompt=system_prompt)

    # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    # url = "https://www.scienceabc.com/wp-content/uploads/2018/09/injured-man.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    image = Image.open(
        "/home/zz/workspace/CloudTrack/gss1_jpg.rf.3d14b3515ea31e9a0668a282efe6ea6d_0.jpg"
    )
    prompt = "Based on this information, would you recommend sending help? Answer with yes or no."

    ans = llava.run_inference(prompt, image)
    print(ans)

