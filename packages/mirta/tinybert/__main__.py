from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")

model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")

nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

def main(req):
    try:
        context=req.POST.get("context")
        question=req.POST.get("question")
        results=nlp({
              'question': question,
              'context': context    
              })
        
        return {"body": results}
    
    except Exception as E:
        return {"body":str(E)}