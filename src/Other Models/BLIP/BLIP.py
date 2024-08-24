!pip install -qU transformers
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
# load the image we will test BLIP on
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
#image
# load necessary components: the processor and the model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
def get_answer_blip(model, processor, image, question):
    """Answers the given question and handles all the preprocessing and postprocessing steps"""
    # preprocess the given image and question
    inputs = processor(image, question, return_tensors="pt")
    # generate the answer (get output)
    out = model.generate(**inputs)
    # post-process the output to get human friendly english text
    print(processor.decode(out[0], skip_special_tokens=True))
    return
  # sample question 1
question = "how many dogs are in the picture?"
get_answer_blip(model, processor, image, question)
# sample question 2
question = "how will you describe the picture?"
get_answer_blip(model, processor, image, question)
