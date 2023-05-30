# pip install --upgrade openai
import openai

completion = openai.Completion.create(model="text-davinci-003",  # openai.Model.list()
                                      prompt="write a tagline for Twilio",
                                      temperature=1.8,  # temperature creativeness of response
                                      # top_p=1, #nucleus sampling
                                      max_tokens=3000,  # max tokens
                                      # frequency_penalty=0.5,  # frequency penalty
                                      )

print(completion)
