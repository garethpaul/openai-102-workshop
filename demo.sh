# write curl command to call the completions for openai
curl -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
  "model": "text-davinci-003",
  "prompt": "Write a tag line for Twilio",
  "temperature": 0.7,
  "max_tokens": 150,
  "top_p": 1,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.6
}' \
  https://api.openai.com/v1/completions
