from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

template = PromptTemplate.from_template(
    "Explain why a financial risk model would classify someone with credit score {score} and debt ratio {debt} as high risk."
)
llm = OpenAI(temperature=0.3)
chain = LLMChain(llm=llm, prompt=template)
explanation = chain.run({"score": 640, "debt": 0.6})
print(explanation)
