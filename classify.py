import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import openai
import json
import re
# Load fine-tuned RoBERTa model
roberta_path = "bert/roberta_sentiment_model"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_path).to(device)
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_path)

all_emotions = ["confidence", "satisfaction", "hope", "trust_medical", "anxiety", "anger"]

# Function to classify with RoBERTa
def classify_with_roberta(text):
    roberta_model.eval()
    encoding = roberta_tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)

    with torch.no_grad():
        output = roberta_model(**encoding).logits.sigmoid().cpu().numpy().flatten()

    return {emotion: float(score) for emotion, score in zip(all_emotions, output)}

# Function to classify with LLM (Few-shot or Zero-shot)
openai.api_key = "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"

def parse_llm_output_to_dict(output):
    result = {}
    matches = re.findall(r'(\w+):\s*([\d\.]+)', output)
    for emotion, score in matches:
        result[emotion.strip()] = float(score)
    return result

def classify_with_llm(text, few_shot=False):
    if few_shot:
        prompt = f"""
        Below are examples of Reddit posts and their corresponding sentiment classifications:

        Post: "I'm so happy with my new job!"
        Labels: confidence: 0.9, satisfaction: 1.0, hope: 0.8, trust_medical: 0.0, anxiety: 0.1, anger: 0.0

        Now classify the following Reddit post:

        Post: "{text}"
        Labels (JSON format):
        """
    else:
        prompt = f"""
        Given the Reddit post below, classify the sentiment into confidence, satisfaction, hope, trust_medical, anxiety, and anger. Return a JSON response with probabilities.

        Post: "{text}"
        JSON Output:
        """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an expert in fine-grained sentiment classification."},
                  {"role": "user", "content": prompt}]
    )

    output = response["choices"][0]["message"]["content"]

    # ✅ Ensure output is JSON
    try:
        parsed_output = json.loads(output)
        #print(f"LLM Parsed Output: {parsed_output}")  # Debugging print
        return parsed_output  # Return as structured JSON
    except json.JSONDecodeError:
        #print(f"Error parsing LLM output: {output}")
        return {}  # Return empty dictionary if parsing fails

    return parse_llm_output_to_dict(output)
if __name__ == "__main__":
    text = "Ok Reddit do your thing- I used to be perfectly healthy until two years ago I woke up thinking I had the stomach bug which turned into a years long cascade of the most intense gut issues ever, I couldn’t leave the house or eat or do anything without worrying about crapping my pants. Whelp that fixed itself randomly but in the process of trying to figure out what was wrong I had an endoscopy & colonoscopy and then got MSSA on my face (yuck) and ended up on antibiotics for 6 months because my doctors sucked and couldn’t recognize it or help me so it would go away then come back over and over again and id be begging for help. Long story short I had to change my life plans and move home and I went to the eye doctor thinking I needed glasses because I had headaches, nope my optic nerves were swollen and they suspected IIH from overuse of doxycycline. I dealt with debilitating headaches for months (worse when lying down, rushing sensation, tinnitus) and got referred to a neurologist who gave it some time to hopefully fix itself but after 6 weeks it wasnt better so we did a lumbar puncture but it wasn’t successful the first time so I had to do it a second time with fluoroscopy. At this point itd been 6 weeks from deciding to do an LP and 5 months from ending antibiotics. Whelp opening pressure and all labs came back normal. So now do I just have to hope it truly was from doxy and wont come back? And now since the labs were normal i’ve been told I just have migraines. "

    print("RoBERTa Prediction:", classify_with_roberta(text))
    print("LLM Prediction:", classify_with_llm(text, few_shot=True))
